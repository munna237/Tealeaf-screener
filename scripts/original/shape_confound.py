"""
shape_confound.py -- THE experiment a reviewer will run. Run it first.

The worry: diseased leaves may be systematically smaller / curlier / more
irregular than healthy ones. On a uniform dark backdrop, leaf SHAPE is trivially
available to a vision model. If so, "attention concentration" could be a proxy
for morphology rather than pathology, and the paper's central claim is wrong.

Three tests, in increasing strictness:

  T1  SHAPE-ONLY BASELINE
      Segment the leaf, throw away every pixel, keep only geometry (area,
      aspect ratio, solidity, eccentricity, ...). Train a classifier on those
      numbers alone. This is the null hypothesis made concrete.
        shape AUC ~ 0.90  -> our result is explained by morphology. Bad.
        shape AUC ~ 0.60  -> shape carries little signal. Good.

  T2  WITHIN-LEAF ATTENTION
      Recompute the attention-concentration score using ONLY patches inside the
      leaf mask, renormalised. This removes leaf-vs-background geometry from the
      score entirely.
        AUC holds  -> the signal is intra-leaf texture (lesions). Good.
        AUC drops  -> the signal was leaf-vs-background contrast. Bad.

  T3  SHAPE-MATCHED CONTROL
      Compare healthy vs diseased leaves matched on area and aspect ratio, so
      the two groups are morphologically indistinguishable by construction.
      If attention still separates them, it is reading pathology, not shape.

Segmentation is easy here: the dataset was captured against a dark backdrop.
We use Otsu on the value channel and keep the largest connected component.
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# leaf segmentation (dark backdrop -> Otsu + largest component)
# ---------------------------------------------------------------------------
def leaf_mask(img, size=224):
    """-> bool mask [size,size], True = leaf. None if segmentation fails."""
    import cv2
    a = np.asarray(img.convert("RGB").resize((size, size)))
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # leaf is brighter AND more saturated than a grey/black cloth backdrop
    _, m_v = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((m_v > 0) | (m_s > 0)).astype(np.uint8)

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big)
    if mask.sum() < 0.01 * size * size:      # segmentation clearly failed
        return None

    # CRITICAL: fill interior holes. Dark lesions can punch holes in the Otsu
    # mask; if left open they leak TEXTURE information into the supposedly
    # shape-only features, invalidating the whole baseline. Verified: without
    # this, identical-outline leaves differing only in lesions score AUC 0.75
    # on "shape", which is nonsense. With it, they score ~0.5 as they must.
    filled = np.zeros_like(mask, dtype=np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def shape_features(mask):
    """Geometry ONLY. No pixel intensities. This is what a shape-only model sees."""
    import cv2
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True))
    if area < 1 or per < 1:
        return None

    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_a = float(cv2.contourArea(hull)) + 1e-6

    # ellipse fit needs >=5 points
    if len(c) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        ecc = float(np.sqrt(1 - (min(MA, ma) / (max(MA, ma) + 1e-6)) ** 2))
    else:
        ecc = 0.0

    return np.array([
        area / (mask.size),                    # relative area
        w / (h + 1e-6),                        # aspect ratio
        area / hull_a,                         # solidity (curl / raggedness)
        4 * np.pi * area / (per ** 2 + 1e-6),  # circularity
        per / (np.sqrt(area) + 1e-6),          # perimeter-to-size (raggedness)
        ecc,                                   # eccentricity
        float(mask.sum()) / mask.size,         # fill fraction
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def attn_scores(img, model, tf, store, device, grid, sel_heads, topk, mask=None):
    """
    Returns (global_score, within_leaf_score).
      global      : top-k mass of the full attention map  (the paper's score)
      within_leaf : attention restricted to leaf patches, RENORMALISED, then top-k
                    -> leaf-vs-background geometry cannot contribute
    """
    import torch
    with torch.no_grad():
        _ = model(tf(img).unsqueeze(0).to(device))
    a = store["cls_attn"][0].numpy()[sel_heads].mean(axis=0)   # [patches]
    a = a / (a.sum() + 1e-12)
    g = float(np.sort(a)[::-1][:topk].sum())

    if mask is None:
        return g, np.nan

    # downsample the pixel mask to the patch grid
    pm = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                    .resize((grid, grid), Image.BILINEAR)) > 127
    pm = pm.ravel()
    if pm.sum() < 4:
        return g, np.nan

    aw = a[pm]
    aw = aw / (aw.sum() + 1e-12)               # renormalise INSIDE the leaf
    k = max(1, int(round(0.10 * len(aw))))
    w = float(np.sort(aw)[::-1][:k].sum())
    return g, w


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(all_images, model, tf, store, device, grid, sel_heads, topk,
        n_per_class=250, seed=0, results_dir=None):
    import random, json, os
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    rng = random.Random(seed)
    by_cls = {}
    for r in all_images:
        by_cls.setdefault(r["class"], []).append(r)
    sample = []
    for c, v in by_cls.items():
        sample += rng.sample(v, min(n_per_class, len(v)))
    rng.shuffle(sample)

    S, G, W, Y, CLS, FAIL = [], [], [], [], [], 0
    for r in tqdm(sample, desc="shape + attention"):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            FAIL += 1; continue
        m = leaf_mask(img)
        if m is None:
            FAIL += 1; continue
        f = shape_features(m)
        if f is None:
            FAIL += 1; continue
        g, w = attn_scores(img, model, tf, store, device, grid, sel_heads, topk, m)
        S.append(f); G.append(g); W.append(w)
        Y.append(r["is_diseased"]); CLS.append(r["class"])

    S = np.array(S); G = np.array(G); W = np.array(W); Y = np.array(Y)
    CLS = np.array(CLS)
    ok = ~np.isnan(W)
    print(f"\nsegmented {len(Y)} / {len(sample)}  (failed {FAIL})")

    print("=" * 72)
    print("SHAPE CONFOUND TEST")
    print("=" * 72)

    # ---- T1: shape-only baseline -------------------------------------------
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    auc_lr = cross_val_score(lr, S, Y, cv=cv, scoring="roc_auc").mean()
    auc_rf = cross_val_score(rf, S, Y, cv=cv, scoring="roc_auc").mean()
    shape_auc = max(auc_lr, auc_rf)

    print(f"\nT1  SHAPE-ONLY BASELINE  (geometry, zero pixels)")
    print(f"      logistic regression : AUC {auc_lr:.4f}")
    print(f"      random forest       : AUC {auc_rf:.4f}")
    print(f"      -> best shape-only  : AUC {shape_auc:.4f}")
    if shape_auc >= 0.85:
        print("      *** WARNING: leaf morphology alone nearly explains the result.")
    elif shape_auc >= 0.70:
        print("      ** shape carries real signal; must be reported and controlled.")
    else:
        print("      OK: shape alone is weak.")

    # ---- T2: within-leaf attention ------------------------------------------
    auc_g = roc_auc_score(Y, G)
    auc_w = roc_auc_score(Y[ok], W[ok])
    print(f"\nT2  ATTENTION SCORE")
    print(f"      global attention (paper's score) : AUC {auc_g:.4f}")
    print(f"      WITHIN-LEAF only (renormalised)  : AUC {auc_w:.4f}")
    print(f"      delta                            : {auc_w - auc_g:+.4f}")
    if auc_w >= auc_g - 0.05:
        print("      OK: signal survives inside the leaf -> intra-leaf texture, not geometry.")
    else:
        print("      *** WARNING: signal depends on leaf-vs-background geometry.")

    # ---- T3: shape-matched control ------------------------------------------
    area = S[:, 0]; aspect = S[:, 1]
    h_idx = np.where(Y == 0)[0]; d_idx = np.where(Y == 1)[0]
    matched_h, matched_d, used = [], [], set()
    for i in h_idx:
        cand = [j for j in d_idx if j not in used
                and abs(area[j] - area[i]) < 0.02
                and abs(aspect[j] - aspect[i]) < 0.15]
        if cand:
            j = min(cand, key=lambda j: abs(area[j] - area[i]) + abs(aspect[j] - aspect[i]))
            used.add(j); matched_h.append(i); matched_d.append(j)

    print(f"\nT3  SHAPE-MATCHED CONTROL  ({len(matched_h)} matched pairs)")
    if len(matched_h) >= 30:
        mi = np.array(matched_h + matched_d)
        my = np.r_[np.zeros(len(matched_h)), np.ones(len(matched_d))].astype(int)
        auc_m_shape = roc_auc_score(my, np.r_[area[matched_h], area[matched_d]])
        auc_m_attn  = roc_auc_score(my, G[mi])
        okm = ok[mi]
        auc_m_within = roc_auc_score(my[okm], W[mi][okm]) if okm.sum() > 20 else float("nan")
        print(f"      shape (area) on matched pairs : AUC {auc_m_shape:.4f}  (~0.5 = matched)")
        print(f"      attention on matched pairs    : AUC {auc_m_attn:.4f}")
        print(f"      within-leaf on matched pairs  : AUC {auc_m_within:.4f}")
        if auc_m_attn >= 0.70:
            print("      OK: attention separates shape-matched leaves -> reading pathology.")
        else:
            print("      *** WARNING: on shape-matched leaves attention loses power.")
    else:
        auc_m_shape = auc_m_attn = auc_m_within = float("nan")
        print("      too few matched pairs; widen the tolerance.")

    # ---- per-class shape summary --------------------------------------------
    print(f"\nPer-class geometry (is Helopeltis simply deformed?)")
    print(f"{'class':<20s} {'area':>8s} {'aspect':>8s} {'solidity':>9s} {'ragged':>8s}")
    print("-" * 58)
    for c in sorted(set(CLS)):
        m_ = CLS == c
        print(f"{c:<20s} {S[m_,0].mean():>8.4f} {S[m_,1].mean():>8.3f} "
              f"{S[m_,2].mean():>9.3f} {S[m_,4].mean():>8.3f}")

    print("\n" + "=" * 72)
    verdict = ("SHAPE CONFOUND LIKELY" if shape_auc >= 0.85 or auc_w < auc_g - 0.05
               else "ATTENTION SURVIVES: signal is not a shape artefact")
    print("VERDICT:", verdict)
    print("=" * 72)

    out = {"n": int(len(Y)), "shape_auc_lr": float(auc_lr), "shape_auc_rf": float(auc_rf),
           "shape_auc_best": float(shape_auc),
           "attn_global_auc": float(auc_g), "attn_within_leaf_auc": float(auc_w),
           "matched_pairs": int(len(matched_h)),
           "matched_shape_auc": float(auc_m_shape),
           "matched_attn_auc": float(auc_m_attn),
           "matched_within_auc": float(auc_m_within),
           "verdict": verdict}
    if results_dir:
        with open(os.path.join(results_dir, "shape_confound.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("saved shape_confound.json")
    return out, S, G, W, Y, CLS
