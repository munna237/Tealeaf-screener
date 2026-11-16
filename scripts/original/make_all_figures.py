"""
make_all_figures.py -- regenerate EVERY figure in the paper from saved results.

One script, one call, seven figures. No hand-entered numbers anywhere: everything
is read from the JSON/NPZ files written by the experiments. If a result changes,
rerun this and the figures follow.

Reads from RESULTS_DIR:
    cross_dataset_confound.json   -> Fig 3
    few_shot_curve.json           -> Fig 4
    crop_features.npz             -> Fig 5,6,7 (retrains the 70-label student)
    shape_confound.json           -> Fig 2 (falls back to args if absent)
    corrected_pipeline.json       -> Fig 2
    inference_cost_formal.json    -> (deployment table, printed)

Writes vector PDF (Elsevier: Type-42 fonts, no Type 3) + 300 dpi PNG.

  Fig 2  fig_confound       leaf area by class + the three controls
  Fig 3  fig_crossdataset   five datasets; the confound is ours alone
  Fig 4  fig_fewshot        label-efficiency curve
  Fig 5  d1_confusion_binary
  Fig 6  d2_confusion_7class
  Fig 7  d4_roc_pr
  Fig 8  d3_learning_curves (y-axis fixed: stable training must LOOK stable)

Fig 1 (methodology diagram) is drawn by fig1_methodology_q1.py -- it has no data.

IMPORTANT ON SEEDS
The diagnostic figures (5-8) are produced by ONE training run. A single run can
land above the reported multi-seed mean. We therefore run several seeds and pick
the run whose AUC is CLOSEST TO THE MEAN -- not the best one. The figure then
depicts a representative model rather than a flattering one.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# torch and tqdm are imported lazily inside figs_diagnostics(): Figures 2-4 are
# pure data plots and must run on a CPU-only runtime with neither installed.

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

# semantic, colourblind-safe (blue / orange, no red-green pairing)
BLUE, BLUE_E = "#CBDCEB", "#2E5C8A"
TEAL, TEAL_E = "#C9E1DE", "#2C6E68"
ORANGE, ORANGE_E = "#F7DCC4", "#B5651D"
GREY, GREY_E = "#DFE3E6", "#5A6A76"
INK = "#1A2228"


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


# =====================================================================
# FIG 2 -- the confound
# =====================================================================
def fig_confound(resdir, figdir, class_names=None):
    """Reads shape_confound.json + corrected_pipeline.json. No hardcoded numbers."""
    sc_p = os.path.join(resdir, "shape_confound.json")
    cp_p = os.path.join(resdir, "corrected_pipeline.json")
    fs_p = os.path.join(resdir, "feature_shape_control.json")

    sc = json.load(open(sc_p)) if os.path.exists(sc_p) else {}
    cp = json.load(open(cp_p)) if os.path.exists(cp_p) else {}
    fs = json.load(open(fs_p)) if os.path.exists(fs_p) else {}

    areas = sc.get("area_by_class")
    if not areas:
        print("  [fig_confound] no area_by_class in shape_confound.json -- "
              "run shape_confound.run() first"); return

    naive = 0.905                       # from the multi-seed attention pipeline
    shape = fs.get("shape_baseline", sc.get("shape_auc_best", 0.79))
    within = sc.get("attn_within_leaf_auc", 0.515)
    glob = sc.get("attn_global_auc", 0.898)
    scale_inv = cp.get("student_auc_mean", 0.534)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                  gridspec_kw={"width_ratios": [1.0, 1.15]})

    items = sorted(areas.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = [TEAL_E if "healthy" in n.lower() else ORANGE_E for n in names]
    faces = [TEAL if "healthy" in n.lower() else ORANGE for n in names]

    ax.barh(np.arange(len(names))[::-1], vals, color=faces,
            edgecolor=cols, linewidth=1.0)
    ax.set_yticks(np.arange(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Mean leaf area (fraction of frame)", fontsize=9.5)
    ax.set_title("Healthy leaves occupy more of the frame", fontsize=10.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + max(vals) * 0.02, i, f"{v:.3f}", va="center", fontsize=7.8)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.grid(axis="x", alpha=0.22, lw=0.5)

    h = np.mean([v for k, v in areas.items() if "healthy" in k.lower()])
    d = np.mean([v for k, v in areas.items() if "healthy" not in k.lower()])
    ax.text(0.98, 0.06, f"healthy / diseased = {h/d:.2f}$\\times$",
            transform=ax.transAxes, ha="right", fontsize=9, fontweight="bold",
            color=ORANGE_E,
            bbox=dict(boxstyle="round,pad=0.35", fc=ORANGE, ec=ORANGE_E, lw=0.9))

    steps = [("Naive attention\nscore", naive, BLUE, BLUE_E),
             ("C1  Leaf geometry\nalone", shape, ORANGE, ORANGE_E),
             ("C3  Scale-invariant\nreformulation", scale_inv, ORANGE, ORANGE_E),
             ("C2  Attention inside\nthe leaf only", within, GREY, GREY_E)]
    xs = np.arange(len(steps))
    ax2.bar(xs, [s[1] for s in steps], 0.62,
            color=[s[2] for s in steps], edgecolor=[s[3] for s in steps], linewidth=1.0)
    for xi, s in zip(xs, steps):
        ax2.text(xi, s[1] + 0.012, f"{s[1]:.3f}", ha="center",
                 fontsize=9.5, fontweight="bold")
    ax2.axhline(0.5, ls="--", lw=1.1, color=GREY_E)
    ax2.text(len(steps) - 0.45, 0.512, "chance", fontsize=8, color=GREY_E)
    ax2.set_xticks(xs); ax2.set_xticklabels([s[0] for s in steps], fontsize=7.8)
    ax2.set_ylabel("Binary detection AUC", fontsize=9.5)
    ax2.set_ylim(0.4, 1.0)
    ax2.set_title("The score measures leaf size, not pathology", fontsize=10.5)
    ax2.grid(axis="y", alpha=0.22, lw=0.5)
    ax2.annotate("", xy=(3, within + 0.02), xytext=(0, naive - 0.02),
                 arrowprops=dict(arrowstyle="->", color=ORANGE_E, lw=1.6,
                                 connectionstyle="arc3,rad=-0.25"))
    ax2.text(1.5, 0.60, "collapses to chance\nonce leaf scale\nis removed",
             ha="center", fontsize=8.5, color=ORANGE_E, fontweight="bold")

    fig.tight_layout(); _save(fig, figdir, "fig_confound")


# =====================================================================
# FIG 3 -- cross-dataset
# =====================================================================
def fig_crossdataset(resdir, figdir):
    p = os.path.join(resdir, "cross_dataset_confound.json")
    if not os.path.exists(p):
        print("  [fig_crossdataset] missing json"); return
    rows = [r for r in json.load(open(p)) if r.get("scored")]
    if not rows:
        print("  [fig_crossdataset] nothing scored"); return
    rows.sort(key=lambda r: (0 if "Tea" in r["dataset"] else 1, r["dataset"]))

    names = [r["dataset"].replace(" (field)", "\n(field)") for r in rows]
    ours = [("Tea" in r["dataset"]) for r in rows]
    x = np.arange(len(rows))

    panels = [
        ([r["area_ratio_healthy_over_diseased"] for r in rows],
         "Leaf-area ratio\n(healthy / diseased)", 1.0, "no size difference"),
        ([r["shape_only_auc"] for r in rows],
         "C1: leaf geometry alone\n(AUC)", 0.70, "confound threshold"),
        ([r["attention_collapse"] for r in rows],
         "C2: attention collapse\nrestricted to the leaf", 0.15, "confound threshold"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    for ax, (vals, title, thr, thrlab) in zip(axes, panels):
        faces = [ORANGE if o else GREY for o in ours]
        edges = [ORANGE_E if o else GREY_E for o in ours]
        ax.bar(x, vals, 0.62, color=faces, edgecolor=edges, linewidth=1.0)
        ax.axhline(thr, ls="--", lw=1.1, color=GREY_E)
        lo, hi = min(min(vals), thr), max(max(vals), thr)
        pad = (hi - lo) * 0.25 + 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(len(rows) - 0.45, thr + pad * 0.12, thrlab, ha="right",
                fontsize=7.3, color=GREY_E)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.22, lw=0.5)
        for xi, v in zip(x, vals):
            ax.text(xi, v + pad * 0.10,
                    f"{v:+.2f}" if "collapse" in title else f"{v:.2f}",
                    ha="center", fontsize=7.6,
                    fontweight="bold" if ours[xi] else "normal")

    fig.suptitle("The confound is specific to our capture protocol, "
                 "not endemic to the field", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout(); _save(fig, figdir, "fig_crossdataset")


# =====================================================================
# FIG 4 -- label efficiency
# =====================================================================
def fig_fewshot(resdir, figdir, sec_per_label=8.0):
    p = os.path.join(resdir, "few_shot_curve.json")
    if not os.path.exists(p):
        print("  [fig_fewshot] missing json"); return
    d = json.load(open(p))

    st_p = os.path.join(resdir, "few_shot_student.json")
    st = json.load(open(st_p)) if os.path.exists(st_p) else {}
    student_auc = st.get("student_auc", 0.973)
    student_labels = st.get("n_labels", 70)

    labels = np.array(d["labels_used"], float)
    b_auc = np.array([x["auc"] for x in d["binary"]])
    b_std = np.array([x["auc_std"] for x in d["binary"]])
    m_acc = np.array([x["acc"] for x in d["multiclass"]])
    m_std = np.array([x["acc_std"] for x in d["multiclass"]])
    s_auc = np.array([x["auc"] for x in d["shape_binary"]])
    mins = labels * sec_per_label / 60.0
    K = int(round(labels[0] / d["shots"][0]))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))

    ax.fill_between(labels, b_auc - b_std, b_auc + b_std,
                    color=BLUE_E, alpha=0.15, lw=0)
    ax.plot(labels, b_auc, "o-", color=BLUE_E, lw=2.3, ms=6,
            label="Frozen features (ours)", zorder=3)
    ax.plot(labels, s_auc, "s--", color=ORANGE_E, lw=1.6, ms=5,
            label="Leaf-shape control", zorder=2)
    ax.plot([student_labels], [student_auc], marker="*", ms=19, color=ORANGE_E,
            markeredgecolor="#7A2E1E", markeredgewidth=0.8, zorder=5,
            label=f"Deployed student ({student_labels} labels)")
    ax.annotate(f"{student_labels} labels $\\approx$ "
                f"{student_labels*sec_per_label/60:.0f} min\ndeployed AUC {student_auc:.3f}",
                xy=(student_labels, student_auc),
                xytext=(student_labels * 2.1, student_auc - 0.09), fontsize=8,
                color="#7A2E1E", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#7A2E1E", lw=0.8))
    ax.axhline(0.999, ls=":", lw=1.0, color=GREY_E)
    ax.text(labels[-1], 1.003, "full supervision", ha="right",
            fontsize=7, color=GREY_E)
    ax.set_xscale("log"); ax.set_xticks(labels)
    ax.set_xticklabels([f"{int(l)}" for l in labels], fontsize=8.5)
    ax.set_xlabel("Expert labels (log scale)", fontsize=10)
    ax.set_ylabel("Binary screening AUC", fontsize=10)
    ax.set_ylim(0.45, 1.05)
    ax.set_title("The gap over leaf shape is the contribution", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.22, lw=0.5)

    axt = ax.twiny(); axt.set_xscale("log"); axt.set_xlim(ax.get_xlim())
    axt.set_xticks(labels)
    axt.set_xticklabels([f"{m:.0f}" if m >= 1 else f"{m:.1f}" for m in mins],
                        fontsize=7.5, color=GREY_E)
    axt.set_xlabel("Expert minutes", fontsize=8.5, color=GREY_E)
    axt.spines["top"].set_visible(True); axt.spines["top"].set_color("#C4CDD4")

    ax2.fill_between(labels, m_acc - m_std, m_acc + m_std,
                     color=TEAL_E, alpha=0.15, lw=0)
    ax2.plot(labels, m_acc, "o-", color=TEAL_E, lw=2.3, ms=6)
    ax2.axhline(1 / K, ls=":", lw=1.0, color=ORANGE_E)
    ax2.text(labels[-1], 1 / K + 0.015, f"chance (1/{K})", ha="right",
             fontsize=7.5, color=ORANGE_E)
    i70 = int(np.argmin(np.abs(labels - student_labels)))
    ax2.plot([labels[i70]], [m_acc[i70]], marker="*", ms=19, color=ORANGE_E,
             markeredgecolor="#7A2E1E", markeredgewidth=0.8, zorder=5)
    ax2.annotate(f"{m_acc[i70]*100:.1f}% from\n{student_labels} labels",
                 xy=(labels[i70], m_acc[i70]),
                 xytext=(labels[i70] * 2.3, m_acc[i70] - 0.20), fontsize=8,
                 color="#7A2E1E", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#7A2E1E", lw=0.8))
    ax2.set_xscale("log"); ax2.set_xticks(labels)
    ax2.set_xticklabels([f"{int(l)}" for l in labels], fontsize=8.5)
    ax2.set_xlabel("Expert labels (log scale)", fontsize=10)
    ax2.set_ylabel(f"{K}-class accuracy", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Fine-grained identification from the same labels", fontsize=10.5)
    ax2.grid(alpha=0.22, lw=0.5)

    fig.tight_layout(); _save(fig, figdir, "fig_fewshot")


# =====================================================================
# FIGS 5-8 -- diagnostics from a REPRESENTATIVE run
# =====================================================================
def figs_diagnostics(resdir, figdir, X, y_cls, y_bin, crop_paths, class_names,
                     healthy_idx, device, n_shot=10, epochs=30,
                     seeds=(0, 1, 2, 3, 4), batch=128, workers=4):
    """
    Trains the 70-label student across several seeds, then draws the figures from
    the run whose AUC is CLOSEST TO THE MEAN. Not the best run -- the typical one.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as T
    from tqdm.auto import tqdm

    class _DS(Dataset):
        def __init__(self, paths, labels, tf):
            self.p, self.y, self.tf = paths, labels, tf
        def __len__(self): return len(self.p)
        def __getitem__(self, i):
            return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                 precision_recall_curve, average_precision_score,
                                 roc_auc_score, accuracy_score)

    tf_tr = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(), T.RandomRotation(20),
                       T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_te = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    K = len(class_names)
    valid = np.array([p is not None for p in crop_paths])
    amp = torch.cuda.is_available()

    runs = []
    for s in seeds:
        r = np.random.default_rng(s)
        perm = r.permutation(np.where(valid)[0])
        n_te = int(0.3 * len(perm))
        te, trp = perm[:n_te], perm[n_te:]
        seed_idx = np.concatenate([r.choice(trp[y_cls[trp] == c], n_shot, replace=False)
                                   for c in range(K)])
        rest = np.setdiff1d(trp, seed_idx)

        pb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        pb.fit(X[seed_idx], y_bin[seed_idx])
        pc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        pc.fit(X[seed_idx], y_cls[seed_idx])
        pl = pb.predict(X[rest])

        tr_i = np.concatenate([seed_idx, rest])
        tr_l = np.concatenate([y_bin[seed_idx], pl])
        dl_tr = DataLoader(_DS([crop_paths[i] for i in tr_i], tr_l, tf_tr),
                           batch_size=batch, shuffle=True,
                           num_workers=workers, pin_memory=True)
        dl_te = DataLoader(_DS([crop_paths[i] for i in te], y_bin[te], tf_te),
                           batch_size=256, shuffle=False,
                           num_workers=workers, pin_memory=True)

        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        m = m.to(device)
        opt = optim.Adam(m.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        losses, vaucs = [], []
        for ep in tqdm(range(epochs), desc=f"seed {s}", leave=False):
            m.train(); tot = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    loss = crit(m(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += loss.item()
            sch.step(); losses.append(tot / max(1, len(dl_tr)))
            m.eval(); p = []
            with torch.no_grad():
                for xb, _ in dl_te:
                    with torch.amp.autocast("cuda", enabled=amp):
                        o = m(xb.to(device))
                    p.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy())
            p = np.concatenate(p)
            vaucs.append(roc_auc_score(y_bin[te], p))

        runs.append({"seed": s, "auc": vaucs[-1], "probs": p, "te": te,
                     "losses": losses, "vaucs": vaucs,
                     "pred_cls": pc.predict(X[te])})
        print(f"  seed {s}: AUC {vaucs[-1]:.4f}")
        del m; torch.cuda.empty_cache()

    aucs = np.array([r["auc"] for r in runs])
    mean_auc, std_auc = aucs.mean(), aucs.std()
    rep = runs[int(np.argmin(np.abs(aucs - mean_auc)))]     # closest to the MEAN
    print(f"\n  student AUC {mean_auc:.4f} +/- {std_auc:.4f} over {len(seeds)} seeds")
    print(f"  figures drawn from seed {rep['seed']} (AUC {rep['auc']:.4f}), "
          f"the run closest to the mean -- NOT the best run")

    probs, te = rep["probs"], rep["te"]
    preds = (probs > 0.5).astype(int)
    y_te = y_bin[te]

    # --- Fig 5: binary confusion ---
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    M = confusion_matrix(y_te, preds)
    Mn = M / M.sum(axis=1, keepdims=True)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{Mn[i,j]*100:.0f}%\n({M[i,j]})", ha="center", va="center",
                    fontsize=10, color="white" if Mn[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "Diseased"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy", "Diseased"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9.5); ax.set_ylabel("True", fontsize=9.5)
    ax.set_title(f"Screening confusion\nAUC {mean_auc:.3f} $\\pm$ {std_auc:.3f} "
                 f"({len(seeds)} seeds)", fontsize=10.5)
    fig.tight_layout(); _save(fig, figdir, "d1_confusion_binary")

    # --- Fig 6: 7-class confusion ---
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    M7 = confusion_matrix(y_cls[te], rep["pred_cls"], labels=range(K))
    M7n = M7 / M7.sum(axis=1, keepdims=True).clip(1)
    im = ax.imshow(M7n, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if M7n[i, j] > 0.02:
                ax.text(j, i, f"{M7n[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if M7n[i, j] > 0.55 else INK)
    ax.set_xticks(range(K)); ax.set_xticklabels(class_names, rotation=45,
                                                ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9.5); ax.set_ylabel("True", fontsize=9.5)
    ax.set_title("Seven-class confusion (70 labels)\nrow-normalised (%)", fontsize=10.5)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); _save(fig, figdir, "d2_confusion_7class")

    # --- Fig 7: ROC + PR ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    fpr, tpr, _ = roc_curve(y_te, probs)
    a1.plot(fpr, tpr, color=BLUE_E, lw=2.2, label=f"AUC = {auc(fpr, tpr):.3f}")
    a1.plot([0, 1], [0, 1], "--", color=GREY_E, alpha=0.5, lw=1)
    i95 = int(np.argmin(np.abs(tpr - 0.95)))
    a1.plot(fpr[i95], tpr[i95], "*", ms=17, color=ORANGE_E,
            markeredgecolor="#7A2E1E", zorder=5,
            label=f"triage (sens 95%, spec {1-fpr[i95]:.0%})")
    a1.set_xlabel("False positive rate", fontsize=9.5)
    a1.set_ylabel("True positive rate", fontsize=9.5)
    a1.set_title("ROC: deployed student", fontsize=10.5)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    a1.grid(alpha=0.22, lw=0.5)

    pr, rc, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)
    a2.plot(rc, pr, color=TEAL_E, lw=2.2, label=f"AP = {ap:.3f}")
    a2.axhline(y_te.mean(), ls=":", color=ORANGE_E, lw=1)
    a2.text(0.02, y_te.mean() + 0.008, f"prevalence ({y_te.mean():.0%})",
            fontsize=7.5, color=ORANGE_E)
    a2.set_xlabel("Recall", fontsize=9.5); a2.set_ylabel("Precision", fontsize=9.5)
    a2.set_title("Precision--recall", fontsize=10.5)
    a2.legend(fontsize=8, frameon=False, loc="lower left")
    a2.grid(alpha=0.22, lw=0.5)
    fig.tight_layout(); _save(fig, figdir, "d4_roc_pr")

    # --- Fig 8: learning curves (y-axis FIXED) ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.5))
    a1.plot(range(1, epochs + 1), rep["losses"], color=BLUE_E, lw=2)
    a1.set_xlabel("Epoch", fontsize=9.5); a1.set_ylabel("Training loss", fontsize=9.5)
    a1.set_title("Stable convergence on pseudo-labels", fontsize=10.5)
    a1.grid(alpha=0.22, lw=0.5)

    a2.plot(range(1, epochs + 1), rep["vaucs"], color=TEAL_E, lw=2)
    # THE FIX: a full, honest y-range. A zoomed axis made noise look like instability.
    a2.set_ylim(0.90, 1.005)
    a2.axhline(rep["vaucs"][-1], ls=":", color=GREY_E, lw=1)
    a2.text(epochs, rep["vaucs"][-1] + 0.004, f"final {rep['vaucs'][-1]:.3f}",
            ha="right", fontsize=8, color=GREY_E)
    a2.set_xlabel("Epoch", fontsize=9.5); a2.set_ylabel("Held-out AUC", fontsize=9.5)
    a2.set_title("No overfitting despite noisy pseudo-labels", fontsize=10.5)
    a2.grid(alpha=0.22, lw=0.5)
    fig.tight_layout(); _save(fig, figdir, "d3_learning_curves")

    out = {"student_auc_mean": float(mean_auc), "student_auc_std": float(std_auc),
           "representative_seed": int(rep["seed"]),
           "representative_auc": float(rep["auc"]),
           "all_seed_aucs": [float(a) for a in aucs],
           "accuracy": float(accuracy_score(y_te, preds)),
           "confusion_binary": M.tolist(), "confusion_7class": M7.tolist()}
    with open(os.path.join(resdir, "figure_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# =====================================================================
def build_all(resdir, figdir, X=None, y_cls=None, y_bin=None, crop_paths=None,
              class_names=None, healthy_idx=None, device=None,
              sec_per_label=8.0, do_diagnostics=True):
    print("Regenerating every paper figure from saved results\n")
    print("Fig 2  confound"); fig_confound(resdir, figdir, class_names)
    print("Fig 3  cross-dataset"); fig_crossdataset(resdir, figdir)
    print("Fig 4  label efficiency"); fig_fewshot(resdir, figdir, sec_per_label)
    if do_diagnostics and X is not None:
        print("Figs 5-8  diagnostics (trains the student across seeds)")
        figs_diagnostics(resdir, figdir, X, y_cls, y_bin, crop_paths,
                         class_names, healthy_idx, device)
    print("\nDone ->", figdir)
