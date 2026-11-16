"""
make_results_figures.py -- builds the Results figures from the saved JSONs.

Run in Colab after the experiments. Outputs vector PDF + PNG to FIGURES_DIR.
Every number is read from disk, so figures regenerate correctly when results change.

Usage in Colab:
    import make_results_figures as mrf
    mrf.build_all(RESULTS_DIR, FIGURES_DIR)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42      # embed real fonts (Elsevier)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

C_MAIN = "#2C5F8A"     # attention / ours
C_BASE = "#B8C4CE"     # baselines
C_WARN = "#C8553D"     # emphasis
C_OK   = "#4A7C59"


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


# ---------------------------------------------------------------------------
# FIG: localisation vs null models  (the make-or-break claim)
# ---------------------------------------------------------------------------
def fig_localisation(resdir, figdir):
    p = os.path.join(resdir, "localization_validation.json")
    if not os.path.exists(p):
        print("skip localisation (no json)"); return
    r = json.load(open(p))

    metrics = ["pointing_game", "attn_mass_in", "iou@top10", "iou@top20"]
    labels  = ["Pointing\ngame", "Attention mass\nin lesion", "IoU@10%", "IoU@20%"]
    methods = [("attention", "Attention (ours)", C_MAIN),
               ("center",    "Center prior",     C_BASE),
               ("gt_prior",  "Dataset prior",    "#8FA3B0"),
               ("random",    "Random",           "#DCE3E8")]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    x = np.arange(len(metrics)); w = 0.2
    for i, (k, name, col) in enumerate(methods):
        vals = [r[k][m] for m in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=col,
               edgecolor="#33414B", linewidth=0.6)

    # chance line for mass-in-GT
    ax.axhline(r["gt_area_frac"], ls=":", lw=1.2, color=C_WARN, zorder=0)
    ax.text(len(metrics) - 0.4, r["gt_area_frac"] + 0.012,
            f"chance ({r['gt_area_frac']*100:.1f}%)", fontsize=7.5, color=C_WARN)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Lesion localisation vs. null models  (n={r['n']})", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    # per-class pointing game -- the morphology story
    pc = r.get("attention_pointing_per_class", {})
    if pc:
        items = sorted(pc.items(), key=lambda kv: kv[1])
        names = [k for k, _ in items]; vals = [v for _, v in items]
        cols = [C_OK if v >= 0.9 else (C_WARN if v < 0.78 else C_MAIN) for v in vals]
        ax2.barh(np.arange(len(names)), vals, color=cols,
                 edgecolor="#33414B", linewidth=0.6)
        ax2.set_yticks(np.arange(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Pointing-game accuracy", fontsize=9)
        ax2.set_title("Focal lesions localise best", fontsize=10.5)
        ax2.grid(axis="x", alpha=0.25, lw=0.6)
        for i, v in enumerate(vals):
            ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7.5)

    fig.tight_layout()
    _save(fig, figdir, "fig_localisation")


# ---------------------------------------------------------------------------
# FIG: screening utility -- the agricultural claim
# ---------------------------------------------------------------------------
def fig_triage(resdir, figdir):
    p = os.path.join(resdir, "triage_analysis.json")
    if not os.path.exists(p):
        print("skip triage (no json)"); return
    d = json.load(open(p))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    # (a) sensitivity/workload trade-off at dataset prevalence
    rows = d["dataset_prevalence"]
    sens = [r["target_sensitivity"] * 100 for r in rows]
    clr  = [r["workload_reduction"] * 100 for r in rows]
    miss = [r["missed_disease_rate"] * 100 for r in rows]
    ax.plot(sens, clr, "o-", color=C_MAIN, lw=2, ms=6, label="Images auto-cleared")
    ax.plot(sens, miss, "s--", color=C_WARN, lw=1.6, ms=5, label="Diseased leaves missed")
    ax.set_xlabel("Target sensitivity (%)", fontsize=9.5)
    ax.set_ylabel("% of images", fontsize=9.5)
    ax.set_title("Operating-point trade-off\n(dataset prevalence)", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)

    # (b) workload removed vs FIELD prevalence -- the deployment number
    fp = d["field_prevalence"]
    prev = sorted(float(k) for k in fp)
    cleared = [fp[str(p_)]["workload_reduction"] * 100 for p_ in prev]
    npv     = [fp[str(p_)]["npv_of_cleared"] * 100 for p_ in prev]
    xs = np.arange(len(prev))

    b = ax2.bar(xs, cleared, 0.55, color=C_OK, edgecolor="#33414B", linewidth=0.6)
    for i, (c, n) in enumerate(zip(cleared, npv)):
        ax2.text(i, c + 1.2, f"{c:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i, c / 2, f"NPV\n{n:.1f}%", ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{int(p_*100)}%" for p_ in prev], fontsize=9)
    ax2.set_xlabel("Field prevalence (diseased leaves)", fontsize=9.5)
    ax2.set_ylabel("Expert review workload removed (%)", fontsize=9.5)
    ax2.set_title("Screening utility at 95% sensitivity", fontsize=10.5)
    ax2.set_ylim(0, max(cleared) * 1.25)
    ax2.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    _save(fig, figdir, "fig_triage")


# ---------------------------------------------------------------------------
# FIG: multi-class cluster composition
# ---------------------------------------------------------------------------
def fig_clusters(resdir, figdir, feats_npy="dino_features.npy",
                 labels_npy="dino_labels.npy", class_names=None):
    fp = os.path.join(resdir, feats_npy)
    lp = os.path.join(resdir, labels_npy)
    if not (os.path.exists(fp) and os.path.exists(lp)):
        print("skip clusters (no features)"); return
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = np.load(fp); y = np.load(lp)
    K = int(y.max()) + 1
    names = class_names or [f"c{i}" for i in range(K)]

    Z = PCA(n_components=128, whiten=True, svd_solver="full").fit_transform(X)
    km = KMeans(K, n_init=200, random_state=0).fit(Z)

    M = np.zeros((K, K), dtype=int)
    for t, p_ in zip(y, km.labels_):
        M[p_, t] += 1
    Mn = M / M.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"cluster {i}" for i in range(K)], fontsize=8)
    ax.set_xlabel("True class", fontsize=9.5)
    ax.set_title("Unsupervised cluster composition\n(row-normalised; 7 labels used for naming)",
                 fontsize=10.5)
    for i in range(K):
        for j in range(K):
            if Mn[i, j] > 0.04:
                ax.text(j, i, f"{Mn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if Mn[i, j] > 0.55 else "#22303A")
    fig.colorbar(im, fraction=0.046, label="% of cluster")
    fig.tight_layout()
    _save(fig, figdir, "fig_clusters")


# ---------------------------------------------------------------------------
# FIG: deployment Pareto
# ---------------------------------------------------------------------------
def fig_deploy(resdir, figdir):
    p = os.path.join(resdir, "inference_cost_formal.json")
    if not os.path.exists(p):
        print("skip deploy (no json)"); return
    d = json.load(open(p))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, m in d.items():
        ours = "MobileNet" in name
        ax.scatter(m["cpu_ms"], m["params_M"],
                   s=240 if ours else 130,
                   c=C_MAIN if ours else C_BASE,
                   edgecolor="#33414B", linewidth=0.9, zorder=3,
                   marker="*" if ours else "o")
        ax.annotate(f"{name}\n{m['size_MB']} MB",
                    (m["cpu_ms"], m["params_M"]),
                    textcoords="offset points", xytext=(9, 6),
                    fontsize=7.8, fontweight="bold" if ours else "normal")
    ax.set_xlabel("CPU latency (ms/image)", fontsize=9.5)
    ax.set_ylabel("Parameters (millions)", fontsize=9.5)
    ax.set_title("Deployment cost: the student never ships the teacher", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, figdir, "fig_deploy")


def build_all(resdir, figdir, class_names=None):
    fig_localisation(resdir, figdir)
    fig_triage(resdir, figdir)
    fig_clusters(resdir, figdir, class_names=class_names)
    fig_deploy(resdir, figdir)
    print("\nDone. Figures in", figdir)
