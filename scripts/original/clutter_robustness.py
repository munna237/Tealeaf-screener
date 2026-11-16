"""
clutter_robustness.py -- how far does the deployed model survive outside the
capture protocol?

WHY THIS MATTERS FOR THE PAPER
Field validation is not possible (flooding in Sylhet). The next-best evidence is
to simulate departure from the capture protocol and measure exactly where the
system breaks. This turns "our method needs a dark backdrop" from a caveat into a
quantified operating specification a farmer can follow.

WHAT WE DO
Composite realistic field textures over each test image at increasing opacity
alpha in {0, 0.1, 0.2, 0.3, 0.4, 0.6}, and evaluate:
    - the DEPLOYED 70-label pipeline (leaf-crop -> frozen features -> probe)
    - the leaf-shape control
    - segmentation success rate  (does the leaf still segment at all?)

alpha = 0    : the capture protocol, followed correctly
alpha ~ 0.2  : a slightly imperfect backdrop (cloth with texture, shadows)
alpha ~ 0.4+ : the protocol effectively abandoned

The number the paper needs is the alpha at which accuracy falls below a usable
threshold. That is the operating limit, stated honestly.

NOTE: alpha here degrades the WHOLE image, including the leaf, so it is a harsher
test than a merely cluttered background. We say so, rather than pretending it is
a faithful field simulation. It bounds the failure, it does not replace a field
trial.
"""

import os
import numpy as np
from PIL import Image


def field_textures(n=40, size=224, seed=0):
    """Synthetic soil/foliage textures. Deterministic, so results reproduce."""
    import cv2
    rng = np.random.RandomState(seed)
    out = []
    palettes = [
        [(34, 85, 34), (60, 110, 45), (20, 60, 25)],     # foliage
        [(85, 65, 34), (110, 90, 55), (60, 45, 25)],     # soil
        [(45, 90, 45), (70, 120, 60), (30, 70, 35)],     # grass
        [(70, 60, 40), (95, 85, 60), (50, 40, 30)],      # dry litter
    ]
    for i in range(n):
        pal = palettes[i % len(palettes)]
        img = np.zeros((size, size, 3), np.uint8)
        img[:] = pal[0]
        for _ in range(70):
            x, y = rng.randint(0, size - 20, 2)
            w, h = rng.randint(15, 70, 2)
            col = pal[rng.randint(len(pal))]
            col = np.clip(np.array(col) + rng.randint(-30, 30, 3), 0, 255)
            img[y:y + h, x:x + w] = col
        img = cv2.GaussianBlur(img, (9, 9), 3)
        out.append(Image.fromarray(img))
    return out


def blend(img, bg, alpha):
    if alpha <= 0:
        return img
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    b = np.asarray(bg.resize(img.size)).astype(np.float32)
    return Image.fromarray(((1 - alpha) * a + alpha * b).clip(0, 255).astype(np.uint8))


def run(records, model, tf, device, mask_fn, crop_fn, shape_fn,
        y_bin, y_cls, class_names, healthy_idx,
        alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.6),
        n_shot=10, n_eval=1500, n_repeats=3, seed=0,
        results_dir=None, verbose=True):
    """
    Trains the 70-label probe on CLEAN images (as it would be in practice), then
    evaluates it on progressively cluttered images.
    """
    import torch
    from tqdm import tqdm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score

    rng = np.random.default_rng(seed)
    K = len(class_names)
    bgs = field_textures()
    N = len(records)

    def feats_at(idx, alpha, bg_choice):
        """Extract crop-normalised features for records[idx] at clutter alpha."""
        F, seg_ok = [], []
        for j, i in enumerate(idx):
            img = Image.open(records[i]["path"]).convert("RGB")
            if alpha > 0:
                img = blend(img, bgs[bg_choice[j] % len(bgs)], alpha)
            m = mask_fn(img)
            if m is None:
                F.append(None); seg_ok.append(False); continue
            c = crop_fn(img, m)
            if c is None:
                F.append(None); seg_ok.append(False); continue
            with torch.no_grad():
                f = model(tf(c).unsqueeze(0).to(device)).cpu().numpy()[0]
            F.append(f / (np.linalg.norm(f) + 1e-9)); seg_ok.append(True)
        return F, np.array(seg_ok)

    results = {a: {"auc": [], "acc": [], "seg": []} for a in alphas}

    for rep in range(n_repeats):
        r = np.random.default_rng(seed * 100 + rep)

        # --- the expert labels n_shot per class, on CLEAN images ---
        seed_idx = []
        ok = True
        for c in range(K):
            pool = np.where(y_cls == c)[0]
            if len(pool) < n_shot:
                ok = False; break
            seed_idx.append(r.choice(pool, n_shot, replace=False))
        if not ok:
            continue
        seed_idx = np.concatenate(seed_idx)

        # --- held-out evaluation set ---
        rest = np.setdiff1d(np.arange(N), seed_idx)
        eval_idx = r.choice(rest, min(n_eval, len(rest)), replace=False)
        bg_choice = r.integers(0, len(bgs), len(eval_idx))

        # --- train the probe on CLEAN seed images ---
        Fs, _ = feats_at(seed_idx, 0.0, np.zeros(len(seed_idx), int))
        Xs = np.array([f for f in Fs if f is not None])
        ys = np.array([y_bin[i] for f, i in zip(Fs, seed_idx) if f is not None])
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        probe.fit(Xs, ys)

        # --- evaluate at each clutter level ---
        for a in alphas:
            F, seg = feats_at(tqdm(eval_idx, desc=f"rep{rep+1} a={a}",
                                   leave=False), a, bg_choice)
            keep = [k for k, f in enumerate(F) if f is not None]
            if len(keep) < 50:
                results[a]["auc"].append(np.nan)
                results[a]["acc"].append(np.nan)
                results[a]["seg"].append(seg.mean())
                continue
            Xe = np.array([F[k] for k in keep])
            ye = np.array([y_bin[eval_idx[k]] for k in keep])
            p = probe.predict_proba(Xe)[:, 1]
            results[a]["auc"].append(roc_auc_score(ye, p))
            results[a]["acc"].append(accuracy_score(ye, (p > 0.5).astype(int)))
            results[a]["seg"].append(seg.mean())

    # ---- report ----
    if verbose:
        print("\n" + "=" * 74)
        print("CLUTTER ROBUSTNESS OF THE 70-LABEL DEPLOYED MODEL")
        print("  (probe trained on clean images; evaluated under clutter)")
        print("=" * 74)
        print(f"{'alpha':>6s} {'condition':<26s} {'AUC':>15s} {'acc':>8s} {'seg ok':>8s}")
        print("-" * 74)
        labels = {0.0: "protocol followed", 0.1: "slight backdrop texture",
                  0.2: "imperfect backdrop", 0.3: "poor backdrop",
                  0.4: "protocol abandoned", 0.6: "full field clutter"}
        for a in alphas:
            au = np.nanmean(results[a]["auc"]); su = np.nanstd(results[a]["auc"])
            ac = np.nanmean(results[a]["acc"]); sg = np.nanmean(results[a]["seg"])
            print(f"{a:>6.1f} {labels.get(a,''):<26s} "
                  f"{au:>7.4f}+/-{su:.3f} {ac:>8.3f} {sg*100:>7.0f}%")
        print("-" * 74)
        base = np.nanmean(results[alphas[0]]["auc"])
        for a in alphas[1:]:
            au = np.nanmean(results[a]["auc"])
            if au < base - 0.05:
                print(f"  OPERATING LIMIT: performance degrades materially at "
                      f"alpha >= {a} (AUC {au:.3f} vs {base:.3f} clean).")
                break
        else:
            print("  Model is robust across all clutter levels tested.")
        print("  NB: alpha degrades the WHOLE image, leaf included -- a harsher")
        print("      test than background clutter alone. It bounds the failure")
        print("      mode; it does not substitute for a field trial.")
        print("=" * 74)

    out = {str(a): {"auc": float(np.nanmean(results[a]["auc"])),
                    "auc_std": float(np.nanstd(results[a]["auc"])),
                    "acc": float(np.nanmean(results[a]["acc"])),
                    "seg_rate": float(np.nanmean(results[a]["seg"]))}
           for a in alphas}
    if results_dir:
        import json
        with open(os.path.join(results_dir, "clutter_robustness.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("saved clutter_robustness.json")
    return out
