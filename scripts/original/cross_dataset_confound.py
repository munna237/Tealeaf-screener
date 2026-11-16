"""
cross_dataset_confound.py -- does the leaf-scale confound generalise?

THE CLAIM WE NEED TO EARN
We assert that attention-concentration scoring is confounded by leaf scale, and
that because background-controlled capture is the norm in plant-disease datasets,
other work is likely affected. Right now that is an assertion. This script turns
it into evidence (or refutes it).

FOR EACH DATASET WE MEASURE
  1. leaf area by class          -- is healthy systematically larger/smaller?
  2. area-only AUC               -- can leaf size alone predict disease?
  3. shape-only AUC              -- can full geometry alone predict disease?
  4. naive attention AUC         -- what the confounded pipeline would report
  5. within-leaf attention AUC   -- what survives once scale is removed

The signature of the confound is:
     shape-only HIGH  and  naive attention HIGH  and  within-leaf COLLAPSES.

If another dataset shows the same signature, the confound is a field-wide problem
and our contribution is much stronger. If it does not, the confound is specific to
our capture protocol, and we must say so and narrow the claim. Either outcome is
reportable; not checking is not an option.

NOTE ON SEGMENTATION
Otsu-on-dark-backdrop works for our data and for PlantVillage (uniform grey/black).
It will fail on genuinely cluttered field imagery -- and that failure is itself
informative, because a dataset whose leaves cannot be segmented from the background
is a dataset where this confound cannot arise in the same way. We report the
segmentation success rate for every dataset and refuse to score any dataset where
it falls below 70%.
"""

import os
import numpy as np
from PIL import Image


def scan_dataset(root, name, model, tf, store, device, grid, topk,
                 healthy_keywords=("healthy",), n_per_class=250, seed=0,
                 min_seg_rate=0.70, verbose=True):
    """
    root: a directory of class subfolders.
    Returns a dict of the five measurements, or None if segmentation fails.
    """
    import random
    import torch
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import shape_confound as sc

    if not os.path.isdir(root):
        print(f"[{name}] not found at {root} -- skipping")
        return None

    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    if len(classes) < 2:
        print(f"[{name}] fewer than 2 class folders -- skipping")
        return None

    rng = random.Random(seed)
    recs = []
    for c in classes:
        cdir = os.path.join(root, c)
        fs = [f for f in os.listdir(cdir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        is_h = any(k in c.lower() for k in healthy_keywords)
        for f in rng.sample(fs, min(n_per_class, len(fs))):
            recs.append({"path": os.path.join(cdir, f), "class": c,
                         "is_diseased": 0 if is_h else 1})

    if not any(r["is_diseased"] == 0 for r in recs):
        print(f"[{name}] no healthy class found -- skipping")
        return None

    S, G, W, A, Y, C = [], [], [], [], [], []
    n_try = 0
    for r in tqdm(recs, desc=f"{name}"):
        n_try += 1
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            continue
        m = sc.leaf_mask(img)
        if m is None:
            continue
        f = sc.shape_features(m)
        if f is None:
            continue
        g, w = sc.attn_scores(img, model, tf, store, device, grid,
                              [0, 1], topk, m)
        S.append(f); G.append(g); W.append(w); A.append(float(m.mean()))
        Y.append(r["is_diseased"]); C.append(r["class"])

    seg_rate = len(Y) / max(1, n_try)
    if seg_rate < min_seg_rate:
        print(f"[{name}] segmentation succeeded on only {seg_rate*100:.0f}% "
              f"of images -- REFUSING to score (leaves not separable from "
              f"background; the confound cannot be assessed this way).")
        return {"dataset": name, "segmentation_rate": float(seg_rate),
                "scored": False}

    S = np.array(S); G = np.array(G); W = np.array(W)
    A = np.array(A); Y = np.array(Y); C = np.array(C)
    ok = ~np.isnan(W)

    def _dir(a):
        return max(a, 1 - a)

    area_auc = _dir(roc_auc_score(Y, A))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    shape_auc = max(cross_val_score(lr, S, Y, cv=cv, scoring="roc_auc").mean(),
                    cross_val_score(rf, S, Y, cv=cv, scoring="roc_auc").mean())
    naive_auc = _dir(roc_auc_score(Y, G))
    within_auc = _dir(roc_auc_score(Y[ok], W[ok])) if ok.sum() > 20 else np.nan

    # area by class
    by_cls = {c: float(A[C == c].mean()) for c in sorted(set(C.tolist()))}
    h_cls = [c for c in by_cls if any(k in c.lower() for k in healthy_keywords)]
    h_area = np.mean([by_cls[c] for c in h_cls]) if h_cls else np.nan
    d_area = np.mean([v for c, v in by_cls.items() if c not in h_cls])
    ratio = h_area / (d_area + 1e-9)

    out = {"dataset": name, "n": int(len(Y)), "n_classes": len(set(C.tolist())),
           "segmentation_rate": float(seg_rate), "scored": True,
           "healthy_area": float(h_area), "diseased_area": float(d_area),
           "area_ratio_healthy_over_diseased": float(ratio),
           "area_only_auc": float(area_auc),
           "shape_only_auc": float(shape_auc),
           "naive_attention_auc": float(naive_auc),
           "within_leaf_attention_auc": float(within_auc),
           "attention_collapse": float(naive_auc - within_auc),
           "area_by_class": by_cls}

    if verbose:
        print(f"\n  --- {name} ---")
        print(f"    images scored          : {len(Y)}  "
              f"(segmentation {seg_rate*100:.0f}%)")
        print(f"    healthy leaf area      : {h_area:.4f}")
        print(f"    diseased leaf area     : {d_area:.4f}")
        print(f"    ratio healthy/diseased : {ratio:.2f}x")
        print(f"    area-only AUC          : {area_auc:.4f}")
        print(f"    shape-only AUC         : {shape_auc:.4f}")
        print(f"    naive attention AUC    : {naive_auc:.4f}")
        print(f"    within-leaf attention  : {within_auc:.4f}")
        print(f"    -> attention collapse  : {naive_auc - within_auc:+.4f}")
        conf = (shape_auc >= 0.70 and (naive_auc - within_auc) >= 0.15)
        print(f"    CONFOUND SIGNATURE     : "
              f"{'PRESENT' if conf else 'not present'}")
    return out


def summarise(rows):
    rows = [r for r in rows if r and r.get("scored")]
    if not rows:
        print("nothing scored")
        return
    print("\n" + "=" * 92)
    print("CROSS-DATASET CONFOUND REPLICATION")
    print("=" * 92)
    print(f"{'dataset':<18s} {'n':>6s} {'area ratio':>11s} {'shape':>7s} "
          f"{'naive':>7s} {'within':>7s} {'collapse':>9s} {'confound':>10s}")
    print("-" * 92)
    for r in rows:
        conf = (r["shape_only_auc"] >= 0.70 and r["attention_collapse"] >= 0.15)
        print(f"{r['dataset']:<18s} {r['n']:>6d} "
              f"{r['area_ratio_healthy_over_diseased']:>10.2f}x "
              f"{r['shape_only_auc']:>7.3f} {r['naive_attention_auc']:>7.3f} "
              f"{r['within_leaf_attention_auc']:>7.3f} "
              f"{r['attention_collapse']:>+9.3f} "
              f"{'PRESENT' if conf else '-':>10s}")
    print("-" * 92)
    print("  confound signature = shape-only AUC >= 0.70  AND  attention")
    print("  collapses by >= 0.15 once restricted to the leaf interior.")
    print("=" * 92)
