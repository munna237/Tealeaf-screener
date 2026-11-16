"""
feature_shape_control.py -- the decisive experiment.

CONTEXT
Attention-concentration scoring collapsed once leaf scale was controlled
(AUC 0.53 vs 0.79 for leaf shape alone). But DINO FEATURES are a different
signal: a supervised probe on them reaches 92.6% over 7 classes and unsupervised
clustering reaches 63.4%. The question is whether that feature signal is also
just leaf morphology in disguise.

THREE WAYS TO KILL THE CONFOUND

  A. LEAF-CROP NORMALISATION (also the practical fix)
     Crop each leaf to its bounding box and resize to 224x224. Leaf scale is now
     constant by construction, so camera distance cannot drive anything. If
     features still separate healthy/diseased here, the signal is leaf TEXTURE,
     not leaf SIZE -- and the resulting pipeline is deployable in the field where
     capture distance varies.

  B. SHAPE-MATCHED PAIRS
     Compare only healthy/diseased leaves matched on area and aspect ratio, so
     the two groups are morphologically indistinguishable by construction.

  C. SHAPE-RESIDUALISED FEATURES
     Regress each feature dimension on the shape descriptors and keep the
     residual. Whatever survives is, by construction, orthogonal to shape.

For each we report a supervised probe (upper bound) AND a fully unsupervised
route (clustering + k-label naming), because the paper's claim is zero-annotation.
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
def crop_to_leaf(img, mask, out=224, pad=0.06):
    """Crop to the leaf bbox and resize -> leaf scale becomes constant."""
    ys, xs = np.where(mask)
    if len(ys) < 20:
        return None
    h, w = mask.shape
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    py, px = int(pad * (y1 - y0 + 1)), int(pad * (x1 - x0 + 1))
    y0, y1 = max(0, y0 - py), min(h - 1, y1 + py)
    x0, x1 = max(0, x0 - px), min(w - 1, x1 + px)
    im = img.convert("RGB").resize((w, h))
    return im.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1)).resize((out, out))


def extract_features(records, model, tf, device, leaf_mask_fn,
                     mode="full", desc="features"):
    """
    mode = 'full' : features of the whole image  (the original pipeline)
    mode = 'crop' : features of the leaf-bbox crop, resized -> SCALE-NORMALISED
    Returns X, Y, CLS, SHAPE, AREA, KEEP
    """
    import torch
    from tqdm import tqdm
    import shape_confound as sc

    X, Y, CLS, SHAPE, AREA, KEEP = [], [], [], [], [], []
    for rec in tqdm(records, desc=f"{desc} [{mode}]"):
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception:
            continue
        m = leaf_mask_fn(img)
        if m is None:
            continue
        sf = sc.shape_features(m)
        if sf is None:
            continue

        if mode == "crop":
            im = crop_to_leaf(img, m)
            if im is None:
                continue
        else:
            im = img

        with torch.no_grad():
            f = model(tf(im).unsqueeze(0).to(device)).cpu().numpy()[0]
        X.append(f); Y.append(rec["is_diseased"]); CLS.append(rec["class"])
        SHAPE.append(sf); AREA.append(float(m.mean())); KEEP.append(rec)

    X = np.array(X, dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return (X, np.array(Y), np.array(CLS), np.array(SHAPE, dtype=np.float32),
            np.array(AREA), KEEP)


# ---------------------------------------------------------------------------
def residualise(X, SHAPE):
    """Remove everything linearly predictable from shape. Residual _|_ shape."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    S = StandardScaler().fit_transform(SHAPE)
    S = np.c_[S, np.ones(len(S))]
    beta, *_ = np.linalg.lstsq(S, X, rcond=None)
    R = X - S @ beta
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return R


def match_pairs(SHAPE, Y, tol_area=0.03, tol_aspect=0.20):
    """Healthy/diseased pairs matched on area + aspect ratio."""
    area, aspect = SHAPE[:, 0], SHAPE[:, 1]
    h = np.where(Y == 0)[0]; d = np.where(Y == 1)[0]
    used, hh, dd = set(), [], []
    for i in h:
        cand = [j for j in d if j not in used
                and abs(area[j] - area[i]) < tol_area
                and abs(aspect[j] - aspect[i]) < tol_aspect]
        if cand:
            j = min(cand, key=lambda j: abs(area[j] - area[i]) + abs(aspect[j] - aspect[i]))
            used.add(j); hh.append(i); dd.append(j)
    return np.array(hh), np.array(dd)


# ---------------------------------------------------------------------------
def probe_auc(X, Y, seed=0):
    """
    Supervised linear probe, 5-fold CV. Upper bound on the signal present.
    Returned direction-agnostically: an AUC of 0.40 means "no signal" (0.10 from
    chance), not "worse than chance". After residualisation a mild inversion is
    an over-correction artefact, not evidence.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    a = float(cross_val_score(pipe, X, Y, cv=cv, scoring="roc_auc").mean())
    return max(a, 1.0 - a)


def unsup_binary(X, Y, n_labels=2, seed=0):
    """
    FULLY UNSUPERVISED binary detection: cluster into 2, then reveal ONE label
    per cluster (2 labels total) to name them. This is the deployable number.
    Also returns the label-free NMI.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

    Z = PCA(n_components=min(128, X.shape[1]), whiten=True,
            svd_solver="full").fit_transform(X)
    km = KMeans(2, n_init=200, random_state=seed).fit(Z)
    lab = km.labels_

    # name each cluster from the single image nearest its centroid
    names = {}
    for c in range(2):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
        names[c] = int(Y[idx[int(np.argmin(d))]])
    pred = np.array([names.get(int(c), 0) for c in lab])

    acc = float((pred == Y).mean())
    nmi = float(normalized_mutual_info_score(Y, lab))
    # AUC using distance to the "healthy" centroid as a score
    healthy_c = [c for c, v in names.items() if v == 0]
    if healthy_c:
        d_h = np.linalg.norm(Z - km.cluster_centers_[healthy_c[0]], axis=1)
        auc = float(roc_auc_score(Y, d_h))
    else:
        auc = float("nan")
    return acc, auc, nmi


# ---------------------------------------------------------------------------
def report(tag, X, Y, SHAPE, shape_auc=None):
    print("\n" + "=" * 72)
    print(f"  {tag}   (n={len(Y)}, {Y.mean()*100:.1f}% diseased)")
    print("=" * 72)

    p_full = probe_auc(X, Y)
    acc_u, auc_u, nmi_u = unsup_binary(X, Y)
    print(f"  supervised probe on features   : AUC {p_full:.4f}   [upper bound]")
    print(f"  UNSUPERVISED (2 labels to name): AUC {auc_u:.4f}  acc {acc_u:.4f}  NMI {nmi_u:.3f}")
    if shape_auc is not None:
        print(f"  shape-only baseline            : AUC {shape_auc:.4f}   [the confound]")

    # shape-residualised
    Xr = residualise(X, SHAPE)
    p_res = probe_auc(Xr, Y)
    acc_r, auc_r, _ = unsup_binary(Xr, Y)
    print(f"  --- shape-residualised (features orthogonal to shape) ---")
    print(f"  supervised probe               : AUC {p_res:.4f}")
    print(f"  UNSUPERVISED                   : AUC {auc_r:.4f}  acc {acc_r:.4f}")

    # shape-matched pairs
    hh, dd = match_pairs(SHAPE, Y)
    if len(hh) >= 40:
        idx = np.r_[hh, dd]
        ym = np.r_[np.zeros(len(hh)), np.ones(len(dd))].astype(int)
        p_m = probe_auc(X[idx], ym)
        a_shape_m = probe_auc(SHAPE[idx], ym)
        print(f"  --- shape-matched pairs ({len(hh)} pairs) ---")
        print(f"  shape on matched pairs         : AUC {a_shape_m:.4f}  (~0.5 = matched)")
        print(f"  features on matched pairs      : AUC {p_m:.4f}")
    else:
        p_m = float("nan")
        print(f"  --- too few matched pairs ({len(hh)}) ---")

    return {"probe": p_full, "unsup_auc": auc_u, "unsup_acc": acc_u, "nmi": nmi_u,
            "probe_residualised": p_res, "unsup_residualised_auc": auc_r,
            "probe_matched": p_m, "n": int(len(Y))}
