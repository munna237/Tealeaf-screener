"""Fig. 6 (label efficiency) from released artefacts: reconciled_numbers.json + regen_tables_8_9.json."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "Inter" in f: font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open(sys.argv[1]))["label_curve_C1"]; S = json.load(open(sys.argv[2]))["table9_shape"]
shots = [1, 2, 5, 10, 20, 50, 100]; lab = np.array([7 * s for s in shots]); mins = lab * 8 / 60
fa = np.array([R[str(s)]["auc"]["mean"] for s in shots]); fsd = np.array([R[str(s)]["auc"]["sd"] for s in shots])
a7 = np.array([R[str(s)]["acc7"]["mean"] for s in shots]); a7sd = np.array([R[str(s)]["acc7"]["sd"] for s in shots])
sa = np.array([S[str(s)]["mean"] for s in shots]); ssd = np.array([S[str(s)]["sd"] for s in shots])
NAVY, ORANGE, RED, GREEN = "#1f3a6e", "#d0691f", "#c0392b", "#1e6b45"
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9))
ax.fill_between(lab, fa - fsd, np.minimum(fa + fsd, 1), color=NAVY, alpha=0.14, lw=0)
ax.fill_between(lab, sa - ssd, sa + ssd, color=ORANGE, alpha=0.14, lw=0)
ax.plot(lab, fa, "-o", color=NAVY, lw=2.2, ms=6.5, label="Frozen features (ours)", zorder=3)
ax.plot(lab, sa, "--s", color=ORANGE, lw=1.8, ms=6.5, label="Leaf-shape control", zorder=3)
ax.axhline(0.999, color="grey", ls=":", lw=1); ax.text(8.2, 1.006, "full supervision (0.999)", color="grey", fontsize=8.5)
ax.plot([70], [0.986], marker="*", ms=17, color=RED, mec="black", mew=0.7, ls="none", label="Deployed student (70 labels)", zorder=4)
ax.annotate("70 labels ≈ 9 min\ndeployed AUC 0.986", xy=(70, 0.986), xytext=(110, 0.885), color=RED, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=RED, lw=1))
ax.set_xscale("log"); ax.set_xticks(lab); ax.set_xticklabels(lab); ax.minorticks_off()
ax.set_ylim(0.3, 1.035); ax.set_xlabel("Expert labels (log scale)"); ax.set_ylabel("Binary screening AUC")
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
top = ax.secondary_xaxis("top"); top.set_xticks(lab); top.set_xticklabels(["%.1f" % m if m < 10 else "%.0f" % m for m in mins], fontsize=8.5)
top.set_xlabel("Expert minutes", fontsize=9.5); top.minorticks_off()
ax.set_title("The gap over leaf shape is the contribution", fontsize=11, pad=8)
bx.fill_between(lab, a7 - a7sd, a7 + a7sd, color=GREEN, alpha=0.16, lw=0)
bx.plot(lab, a7, "-o", color=GREEN, lw=2.2, ms=6.5)
bx.plot([70], [a7[3]], marker="*", ms=17, color=RED, mec="black", mew=0.7, ls="none", zorder=4)
bx.annotate("68.5% from\n70 labels", xy=(70, a7[3]), xytext=(110, 0.52), color=RED, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=RED, lw=1))
bx.axhline(1 / 7, color=ORANGE, ls=":", lw=1.2); bx.text(380, 1 / 7 + 0.012, "chance (1/7)", color=ORANGE, fontsize=8.5)
bx.set_xscale("log"); bx.set_xticks(lab); bx.set_xticklabels(lab); bx.minorticks_off()
bx.set_ylim(0, 1); bx.set_xlabel("Expert labels (log scale)"); bx.set_ylabel("7-class accuracy")
bx.set_title("Fine-grained identification from the same labels", fontsize=11, pad=8)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=300); print("saved", sys.argv[3])
