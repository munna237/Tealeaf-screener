"""
localization_eval.py -- Validate that DINOv1 attention actually localizes disease.

Ground truth: hand-drawn boxes over lesions (normalized xyxy in [0,1]).
Prediction  : the CLS->patch attention map (grid x grid, e.g. 14x14) upsampled.

Metrics
  pointing_game : does the single peak attention patch land inside a GT box?
                  Threshold-free, the standard weak-localization metric.
  iou_at_topk   : binarize attention at the top-k% patches, IoU against GT mask.
                  Reported across several k since it is threshold-dependent.
  attn_mass_in  : fraction of total attention mass falling inside GT boxes.
                  Intuitive: "how much of the model's attention is on the lesion?"

Baselines (ESSENTIAL -- leaves are centered, so localization can look good for
trivial reasons). Every metric is also computed for:
  center  : a fixed Gaussian blob at the image center
  random  : uniform random attention map
  gt_prior: the mean GT mask over the whole annotation set (dataset center bias)
A result only means something if attention beats ALL of these.
"""

import numpy as np


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def boxes_to_mask(boxes, grid):
    """boxes: list of [x0,y0,x1,y1] normalized to [0,1]. -> bool [grid,grid]."""
    m = np.zeros((grid, grid), dtype=bool)
    for (x0, y0, x1, y1) in boxes:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        c0 = int(np.floor(x0 * grid)); c1 = int(np.ceil(x1 * grid))
        r0 = int(np.floor(y0 * grid)); r1 = int(np.ceil(y1 * grid))
        c0 = max(0, min(grid - 1, c0)); c1 = max(c0 + 1, min(grid, c1))
        r0 = max(0, min(grid - 1, r0)); r1 = max(r0 + 1, min(grid, r1))
        m[r0:r1, c0:c1] = True
    return m


# ---------------------------------------------------------------------------
# predictors: attention + baselines
# ---------------------------------------------------------------------------
def center_prior(grid, sigma_frac=0.25):
    """Gaussian blob at image center -- the 'leaf is centered' null model."""
    ax = (np.arange(grid) + 0.5) / grid - 0.5
    xx, yy = np.meshgrid(ax, ax)
    s = sigma_frac
    g = np.exp(-(xx**2 + yy**2) / (2 * s * s))
    return g / g.sum()


def random_map(grid, rng):
    m = rng.random((grid, grid))
    return m / m.sum()


# ---------------------------------------------------------------------------
# metrics for a single image
# ---------------------------------------------------------------------------
def pointing_hit(attn, gt_mask):
    """1 if the argmax attention cell is inside the GT mask."""
    idx = np.unravel_index(np.argmax(attn), attn.shape)
    return float(gt_mask[idx])


def iou_at_topk(attn, gt_mask, k_frac):
    """Binarize attention at top-k% cells, IoU with GT."""
    n = attn.size
    k = max(1, int(round(k_frac * n)))
    thr = np.sort(attn.ravel())[::-1][k - 1]
    pred = attn >= thr
    inter = np.logical_and(pred, gt_mask).sum()
    union = np.logical_or(pred, gt_mask).sum()
    return float(inter) / float(union) if union else 0.0


def attn_mass_inside(attn, gt_mask):
    """Fraction of attention mass inside GT (vs GT area fraction = chance)."""
    a = attn / (attn.sum() + 1e-12)
    return float(a[gt_mask].sum())


# ---------------------------------------------------------------------------
# full evaluation over an annotation set
# ---------------------------------------------------------------------------
def evaluate(records, grid, k_fracs=(0.05, 0.10, 0.20), seed=0):
    """
    records: list of dicts with
        'attn'    : [grid,grid] float attention map
        'boxes'   : list of normalized [x0,y0,x1,y1]
        'class'   : str (optional)
    Returns dict of metrics for attention + baselines.
    """
    rng = np.random.default_rng(seed)
    cp = center_prior(grid)

    gts = [boxes_to_mask(r["boxes"], grid) for r in records]
    gt_area = np.mean([g.mean() for g in gts])          # chance level for mass/pointing

    # dataset-level GT prior (mean mask) -- the strongest trivial baseline
    prior = np.mean([g.astype(float) for g in gts], axis=0)
    prior = prior / (prior.sum() + 1e-12)

    preds = {
        "attention": [r["attn"] for r in records],
        "center":    [cp] * len(records),
        "random":    [random_map(grid, rng) for _ in records],
        "gt_prior":  [prior] * len(records),
    }

    out = {"n": len(records), "gt_area_frac": float(gt_area), "grid": grid}
    for name, maps in preds.items():
        res = {
            "pointing_game": float(np.mean([pointing_hit(a, g) for a, g in zip(maps, gts)])),
            "attn_mass_in":  float(np.mean([attn_mass_inside(a, g) for a, g in zip(maps, gts)])),
        }
        for kf in k_fracs:
            res[f"iou@top{int(kf*100)}"] = float(
                np.mean([iou_at_topk(a, g, kf) for a, g in zip(maps, gts)]))
        out[name] = res

    # per-class pointing game for attention
    per_cls = {}
    for r, g in zip(records, gts):
        c = r.get("class", "?")
        per_cls.setdefault(c, []).append(pointing_hit(r["attn"], g))
    out["attention_pointing_per_class"] = {c: float(np.mean(v)) for c, v in per_cls.items()}
    return out


def print_report(res):
    print("=" * 68)
    print("LOCALIZATION VALIDATION")
    print("=" * 68)
    print(f"Annotated images : {res['n']}   grid: {res['grid']}x{res['grid']}")
    print(f"Mean GT area     : {res['gt_area_frac']*100:.1f}% of image")
    print("  (a random point lands in GT this often by chance)")
    print("-" * 68)
    keys = [k for k in res["attention"] if k != "pointing_game" and k != "attn_mass_in"]
    hdr = f"{'method':<12s} {'pointing':>9s} {'mass_in':>9s} " + " ".join(f"{k:>10s}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for name in ["attention", "center", "gt_prior", "random"]:
        r = res[name]
        row = f"{name:<12s} {r['pointing_game']:>9.3f} {r['attn_mass_in']:>9.3f} "
        row += " ".join(f"{r[k]:>10.3f}" for k in keys)
        print(row)
    print("-" * 68)
    a, c = res["attention"], res["center"]
    print(f"attention - center  : pointing {a['pointing_game']-c['pointing_game']:+.3f}, "
          f"mass_in {a['attn_mass_in']-c['attn_mass_in']:+.3f}")
    print("VERDICT: attention must beat center AND gt_prior to be a real result.")
    print("=" * 68)
    print("\nPer-class pointing game (attention):")
    for c, v in sorted(res["attention_pointing_per_class"].items(), key=lambda x: -x[1]):
        print(f"  {c:<22s} {v:.3f}")


def bootstrap_ci(records, grid, metric="pointing_game", n_boot=2000, seed=0):
    """95% CI on attention-minus-center for the chosen metric (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    cp = center_prior(grid)
    gts = [boxes_to_mask(r["boxes"], grid) for r in records]
    if metric == "pointing_game":
        fa = np.array([pointing_hit(r["attn"], g) for r, g in zip(records, gts)])
        fc = np.array([pointing_hit(cp, g) for g in gts])
    else:
        fa = np.array([attn_mass_inside(r["attn"], g) for r, g in zip(records, gts)])
        fc = np.array([attn_mass_inside(cp, g) for g in gts])
    d = fa - fc
    n = len(d)
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
