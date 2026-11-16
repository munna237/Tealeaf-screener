"""
make_qualitative_figures.py -- all image-based figures, generated from YOUR data.

Produces:
  QF1  attention_gallery       : one row per class -- image | attention | overlay
  QF2  head_comparison         : all 6 heads on one leaf, UAHS-selected boxed
  QF3  localization_overlay    : hand-drawn GT box vs attention peak + top-k mask
  QF4  clutter_progression     : same leaf at alpha = 0, .2, .4, .6 with attention
  QF5  severity_spectrum       : low / mid / high severity per class, with overlay
  QF6  success_failure         : side-by-side of what works (focal) vs fails (diffuse)

All read from the frozen DINO in your session. Vector PDF + 300dpi PNG.

Usage in Colab (after Cell A):
    import make_qualitative_figures as qf
    qf.build_all(dino_v1, tf_dv1, _dv1_store, device, all_images, classes,
                 FIGURES_DIR, DV1_GRID, sel_heads=[0,1])
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

CMAP = "inferno"        # reads well in print and greyscale


# ---------------------------------------------------------------- helpers
def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


def get_attn(model, tf, store, device, img, grid, heads=None):
    """-> (mean_map[grid,grid], all_heads[6,grid,grid])"""
    with torch.no_grad():
        _ = model(tf(img).unsqueeze(0).to(device))
    a = store["cls_attn"][0].numpy()                 # [heads, patches]
    per_head = a.reshape(a.shape[0], grid, grid)
    sel = heads if heads is not None else list(range(a.shape[0]))
    return per_head[sel].mean(axis=0), per_head


def _upsample(m, size):
    return np.array(Image.fromarray(m.astype(np.float32)).resize(
        (size, size), Image.BILINEAR))


def _overlay(ax, img, amap, size=224, alpha=0.55):
    im = img.resize((size, size))
    ax.imshow(im)
    ax.imshow(_upsample(amap, size), cmap=CMAP, alpha=alpha)
    ax.axis("off")


def _pick(all_images, cls, k=0):
    pool = [x for x in all_images if x["class"] == cls]
    return pool[min(k, len(pool) - 1)]


# ---------------------------------------------------------------- QF1
def attention_gallery(model, tf, store, device, all_images, classes,
                      figdir, grid, sel_heads, idx=12):
    """One row per class: original | attention map | overlay."""
    n = len(classes)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.35 * n))
    for r, cls in enumerate(classes):
        rec = _pick(all_images, cls, idx)
        img = Image.open(rec["path"]).convert("RGB")
        amap, _ = get_attn(model, tf, store, device, img, grid, sel_heads)

        axes[r, 0].imshow(img.resize((224, 224))); axes[r, 0].axis("off")
        axes[r, 1].imshow(amap, cmap=CMAP); axes[r, 1].axis("off")
        _overlay(axes[r, 2], img, amap)

        axes[r, 0].set_ylabel(cls, fontsize=8)
        axes[r, 0].text(-0.12, 0.5, cls, rotation=90, fontsize=8.5,
                        fontweight="bold", va="center", ha="center",
                        transform=axes[r, 0].transAxes)
        if r == 0:
            for c, t in enumerate(["Input", "DINO attention", "Overlay"]):
                axes[r, c].set_title(t, fontsize=9.5, fontweight="bold")
    fig.suptitle("Frozen DINO attention across all seven classes\n"
                 "(UAHS-selected heads, no labels used)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save(fig, figdir, "qf1_attention_gallery")


# ---------------------------------------------------------------- QF2
def head_comparison(model, tf, store, device, all_images, figdir, grid,
                    sel_heads, cls="Helopelties", idx=12):
    """All 6 heads on one leaf; UAHS-selected heads outlined."""
    rec = _pick(all_images, cls, idx)
    img = Image.open(rec["path"]).convert("RGB")
    _, per_head = get_attn(model, tf, store, device, img, grid)
    H = per_head.shape[0]

    fig, axes = plt.subplots(1, H + 1, figsize=(2.05 * (H + 1), 2.5))
    axes[0].imshow(img.resize((224, 224))); axes[0].axis("off")
    axes[0].set_title("Input\n" + cls, fontsize=9, fontweight="bold")

    for h in range(H):
        ax = axes[h + 1]
        _overlay(ax, img, per_head[h])
        chosen = h in sel_heads
        ax.set_title(f"Head {h}" + ("\n(UAHS)" if chosen else ""),
                     fontsize=9, fontweight="bold" if chosen else "normal",
                     color="#C8553D" if chosen else "#333333")
        if chosen:
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("#C8553D"); s.set_linewidth(3)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Attention heads specialise. UAHS selects the lesion-responsive ones "
                 "without labels.", fontsize=10.5, fontweight="bold", y=1.06)
    fig.tight_layout()
    _save(fig, figdir, "qf2_head_comparison")


# ---------------------------------------------------------------- QF3
def localization_overlay(model, tf, store, device, figdir, grid, sel_heads,
                         ann_json, n_show=8):
    """Hand-annotated GT boxes vs attention: peak marker + top-10% mask."""
    import json
    from matplotlib.patches import Rectangle
    if not os.path.exists(ann_json):
        print("skip QF3 (no annotations)"); return
    ann = {k: v for k, v in json.load(open(ann_json)).items() if v["boxes"]}
    items = list(ann.items())[:n_show]
    if not items:
        print("skip QF3 (no boxed images)"); return

    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.35 * rows))
    axes = np.atleast_2d(axes)

    topk = max(1, int(0.10 * grid * grid))
    for i, (path, v) in enumerate(items):
        ax = axes[i // cols, i % cols]
        img = Image.open(path).convert("RGB")
        amap, _ = get_attn(model, tf, store, device, img, grid, sel_heads)
        _overlay(ax, img, amap, alpha=0.5)

        # ground-truth boxes (human)
        for (x0, y0, x1, y1) in v["boxes"]:
            ax.add_patch(Rectangle((x0 * 224, y0 * 224),
                                   (x1 - x0) * 224, (y1 - y0) * 224,
                                   fill=False, edgecolor="#39FF14",
                                   linewidth=2.0))
        # attention peak
        pr, pc = np.unravel_index(np.argmax(amap), amap.shape)
        ax.plot((pc + 0.5) * 224 / grid, (pr + 0.5) * 224 / grid,
                marker="x", ms=11, mew=2.6, color="white")
        hit = any(x0 <= (pc + .5) / grid <= x1 and y0 <= (pr + .5) / grid <= y1
                  for (x0, y0, x1, y1) in v["boxes"])
        ax.set_title(f"{v['class']}  {'HIT' if hit else 'MISS'}",
                     fontsize=8.5, fontweight="bold",
                     color="#2E7D32" if hit else "#C62828")
    for j in range(len(items), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Attention vs. human-annotated lesions\n"
                 "green = expert box   x = attention peak   heat = attention",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, figdir, "qf3_localization_overlay")


# ---------------------------------------------------------------- QF4
def clutter_progression(model, tf, store, device, all_images, figdir, grid,
                        sel_heads, cls="Red Rust", idx=12,
                        alphas=(0.0, 0.2, 0.4, 0.6)):
    """Same leaf, rising background clutter, with attention -- shows the limit."""
    rec = _pick(all_images, cls, idx)
    img = Image.open(rec["path"]).convert("RGB").resize((224, 224))

    rng = np.random.RandomState(0)
    bg = rng.randint(0, 255, (28, 28, 3), dtype=np.uint8)
    bg = np.array(Image.fromarray(bg).resize((224, 224), Image.BICUBIC)).astype(float)

    fig, axes = plt.subplots(2, len(alphas), figsize=(2.6 * len(alphas), 5.3))
    for j, a in enumerate(alphas):
        blend = Image.fromarray(
            ((1 - a) * np.array(img).astype(float) + a * bg)
            .clip(0, 255).astype(np.uint8))
        amap, _ = get_attn(model, tf, store, device, blend, grid, sel_heads)
        p = amap / (amap.sum() + 1e-12)
        H = float(-(p * np.log(p + 1e-12)).sum())

        axes[0, j].imshow(blend); axes[0, j].axis("off")
        axes[0, j].set_title(("Original" if a == 0 else f"$\\alpha$ = {a}"),
                             fontsize=9.5, fontweight="bold")
        _overlay(axes[1, j], blend, amap)
        axes[1, j].set_title(f"entropy = {H:.2f}", fontsize=8.5,
                             color="#C8553D" if a > 0 else "#2E7D32")
    fig.suptitle(f"Background clutter degrades attention ({cls})\n"
                 "the capture protocol is load-bearing",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, figdir, "qf4_clutter_progression")


# ---------------------------------------------------------------- QF6
def success_failure(model, tf, store, device, all_images, figdir, grid, sel_heads,
                    good=("Helopelties", "Gray Blight"),
                    bad=("Algal leaf spot", "Red Spider"), n=3):
    """The paper's central finding, made visual: focal works, diffuse fails."""
    rows = [(c, "works") for c in good] + [(c, "fails") for c in bad]
    fig, axes = plt.subplots(len(rows), n * 2, figsize=(2.1 * n * 2, 2.3 * len(rows)))
    for r, (cls, tag) in enumerate(rows):
        for k in range(n):
            rec = _pick(all_images, cls, 5 + k * 40)
            img = Image.open(rec["path"]).convert("RGB")
            amap, _ = get_attn(model, tf, store, device, img, grid, sel_heads)
            axes[r, 2 * k].imshow(img.resize((224, 224))); axes[r, 2 * k].axis("off")
            _overlay(axes[r, 2 * k + 1], img, amap)
        col = "#2E7D32" if tag == "works" else "#C62828"
        axes[r, 0].text(-0.14, 0.5, f"{cls}\n({tag})", rotation=90, fontsize=8.5,
                        fontweight="bold", color=col, va="center", ha="center",
                        transform=axes[r, 0].transAxes)
    fig.suptitle("Lesion morphology determines where the method works\n"
                 "focal, high-contrast lesions (top) vs. diffuse symptoms (bottom)",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    _save(fig, figdir, "qf6_success_failure")


# ---------------------------------------------------------------- driver
def build_all(model, tf, store, device, all_images, classes, figdir, grid,
              sel_heads=(0, 1), ann_json=None):
    sel_heads = list(sel_heads)
    attention_gallery(model, tf, store, device, all_images, classes, figdir, grid, sel_heads)
    head_comparison(model, tf, store, device, all_images, figdir, grid, sel_heads)
    clutter_progression(model, tf, store, device, all_images, figdir, grid, sel_heads)
    success_failure(model, tf, store, device, all_images, figdir, grid, sel_heads)
    if ann_json:
        localization_overlay(model, tf, store, device, figdir, grid, sel_heads, ann_json)
    print("\nQualitative figures ->", figdir)
