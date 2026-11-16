"""
few_shot_student.py -- close the deployment loop.

THE GAP
Our label-efficiency result (AUC 0.982 from 70 labels) comes from a linear probe
on frozen DINO features. But DINO is 21.7M parameters / 86.7 MB. Our deployment
table sells MobileNetV3-Small (1.52M / 6.09 MB). Those two claims do not currently
connect: as stated, deployment would require shipping DINO.

THE FIX
Use the 70-label probe as a teacher to pseudo-label the remaining UNLABELLED
training images, then train MobileNetV3-Small on
    70 true labels  +  the teacher's pseudo-labels on everything else.
Report the STUDENT's test AUC. That is the number the deployed 6 MB artefact
actually achieves, and it is the number the paper should quote.

We also report the two obvious baselines so the reader can see what the
distillation buys:
    (a) MobileNetV3 trained on the 70 true labels ALONE (no pseudo-labels)
    (b) the DINO+probe teacher itself (the upper bound)

Everything runs on leaf-crop-normalised images, so leaf scale cannot drive the
result. The test set is never touched by the expert.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T


class CropDS(Dataset):
    """Serves leaf-crop-normalised images with the given labels."""
    def __init__(self, recs, labels, transform, mask_fn, crop_fn):
        self.recs = recs; self.labels = labels
        self.tf = transform; self.mask_fn = mask_fn; self.crop_fn = crop_fn

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        img = Image.open(self.recs[i]["path"]).convert("RGB")
        m = self.mask_fn(img)
        if m is not None:
            c = self.crop_fn(img, m)
            if c is not None:
                img = c
        return self.tf(img), int(self.labels[i])


def run(X, y_cls, y_bin, KEEP, class_names, healthy_idx,
        mask_fn, crop_fn, device,
        n_shot=10, n_repeats=3, test_frac=0.3, epochs=30,
        results_dir=None, seed=0, verbose=True):
    """
    X    : frozen DINO features of the crop-normalised images (for the teacher)
    KEEP : the image records aligned with X
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    tf_train = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_test = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    K = len(class_names)
    n = len(y_bin)
    out = {"teacher": [], "student_distilled": [], "student_labels_only": []}

    for rep in range(n_repeats):
        r = np.random.default_rng(seed * 100 + rep)
        perm = r.permutation(n)
        n_te = int(test_frac * n)
        te, tr_pool = perm[:n_te], perm[n_te:]

        # ---- the expert labels n_shot images per class ----
        picked = []
        for c in range(K):
            idx = tr_pool[y_cls[tr_pool] == c]
            if len(idx) < n_shot:
                picked = None; break
            picked.append(r.choice(idx, n_shot, replace=False))
        if picked is None:
            continue
        seed_idx = np.concatenate(picked)          # the ONLY human labels
        rest = np.setdiff1d(tr_pool, seed_idx)     # unlabelled remainder

        # ---- TEACHER: linear probe on DINO features, 70 labels ----
        probe = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000))
        probe.fit(X[seed_idx], y_bin[seed_idx])
        t_auc = roc_auc_score(y_bin[te], probe.predict_proba(X[te])[:, 1])
        out["teacher"].append(float(t_auc))

        # ---- teacher pseudo-labels the unlabelled remainder ----
        pl = probe.predict(X[rest])
        pl_acc = float((pl == y_bin[rest]).mean())

        train_idx = np.concatenate([seed_idx, rest])
        train_lab = np.concatenate([y_bin[seed_idx], pl])

        # ---- STUDENT (distilled): 70 true + pseudo-labels ----
        def _train_student(idxs, labs):
            ds = CropDS([KEEP[i] for i in idxs], labs, tf_train, mask_fn, crop_fn)
            dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
            m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
            m = m.to(device)
            opt = optim.Adam(m.parameters(), lr=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            crit = nn.CrossEntropyLoss()
            m.train()
            for ep in range(epochs):
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device).long()
                    loss = crit(m(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                sch.step()
            return m

        def _eval(m):
            ds = CropDS([KEEP[i] for i in te], y_bin[te], tf_test, mask_fn, crop_fn)
            dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
            m.eval(); p = []
            with torch.no_grad():
                for xb, _ in dl:
                    p.append(torch.softmax(m(xb.to(device)), 1)[:, 1].cpu().numpy())
            p = np.concatenate(p)
            return (float(roc_auc_score(y_bin[te], p)),
                    float(accuracy_score(y_bin[te], (p > 0.5).astype(int))),
                    float(f1_score(y_bin[te], (p > 0.5).astype(int))))

        m1 = _train_student(train_idx, train_lab)
        a1, acc1, f11 = _eval(m1)
        out["student_distilled"].append({"auc": a1, "acc": acc1, "f1": f11})
        del m1; torch.cuda.empty_cache()

        # ---- STUDENT (labels only): the 70 labels, no pseudo-labels ----
        m2 = _train_student(seed_idx, y_bin[seed_idx])
        a2, acc2, f12 = _eval(m2)
        out["student_labels_only"].append({"auc": a2, "acc": acc2, "f1": f12})
        del m2; torch.cuda.empty_cache()

        if verbose:
            print(f"  rep {rep+1}/{n_repeats}: teacher {t_auc:.4f} | "
                  f"pseudo-label acc {pl_acc:.4f} | "
                  f"student(distilled) {a1:.4f} | student(70 only) {a2:.4f}")

    # ---- summary ----
    t = np.array(out["teacher"])
    sd = np.array([d["auc"] for d in out["student_distilled"]])
    sl = np.array([d["auc"] for d in out["student_labels_only"]])
    acc = np.mean([d["acc"] for d in out["student_distilled"]])
    f1 = np.mean([d["f1"] for d in out["student_distilled"]])

    if verbose:
        print("\n" + "=" * 74)
        print(f"DEPLOYABLE STUDENT from {n_shot * K} expert labels")
        print("=" * 74)
        print(f"  teacher  (DINO + probe, 86.7 MB) : AUC {t.mean():.4f} +/- {t.std():.4f}")
        print(f"  student  (MobileNetV3, 6.09 MB)  : AUC {sd.mean():.4f} +/- {sd.std():.4f}"
              f"   <- THE DEPLOYED NUMBER")
        print(f"           accuracy {acc:.4f} | F1 {f1:.4f}")
        print(f"  student trained on the {n_shot*K} labels ALONE (no distillation):")
        print(f"                                     AUC {sl.mean():.4f} +/- {sl.std():.4f}")
        print(f"  -> distillation gain: {sd.mean() - sl.mean():+.4f}")
        print("=" * 74)

    res = {"n_shot": n_shot, "n_labels": n_shot * K,
           "teacher_auc": float(t.mean()), "teacher_std": float(t.std()),
           "student_auc": float(sd.mean()), "student_std": float(sd.std()),
           "student_acc": float(acc), "student_f1": float(f1),
           "student_labels_only_auc": float(sl.mean()),
           "distillation_gain": float(sd.mean() - sl.mean())}
    if results_dir:
        import json
        with open(os.path.join(results_dir, "few_shot_student.json"), "w") as f:
            json.dump(res, f, indent=2)
        print("saved few_shot_student.json")
    return res
