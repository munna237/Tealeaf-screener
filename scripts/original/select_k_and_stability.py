"""
select_k_and_stability.py -- close the last two holes in the corrected pipeline.

HOLE 1: k was chosen by looking at the answer.
  k=14 gave the best binary AUC, but we only know that because we peeked at the
  labels. A reviewer will (correctly) call that cherry-picking. We therefore
  select k using criteria computable WITHOUT any labels:
      - silhouette score          (cluster separation)
      - Calinski-Harabasz         (between/within dispersion)
      - Davies-Bouldin (lower=better)
      - inertia elbow (kneedle)
  We report the k these pick, and then report performance AT THAT k. If the
  label-free choice lands near 14, the choice is principled. If it lands
  somewhere else, we report performance at the label-free k -- honestly, even if
  it is worse. That is the whole point of the exercise.

HOLE 2: seed variance was +/- 0.051, which is too wide to quote a headline.
  We raise n_init and run more seeds to see whether it tightens.
"""

import numpy as np


def _pca(X, d=128, seed=0):
    from sklearn.decomposition import PCA
    d = min(d, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=d, whiten=True, svd_solver="full",
               random_state=seed).fit_transform(X)


# ---------------------------------------------------------------------------
# 1. LABEL-FREE k SELECTION
# ---------------------------------------------------------------------------
def select_k(X, k_list=(5, 7, 10, 12, 14, 16, 20, 25, 30),
             seed=0, n_init=50, sample_for_silhouette=4000, verbose=True):
    """
    Choose k using ONLY the features. No labels are touched anywhere in here.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                 davies_bouldin_score)
    rng = np.random.default_rng(seed)
    Z = _pca(X, seed=seed)

    # silhouette is O(n^2); subsample for tractability
    sub = rng.choice(len(Z), min(sample_for_silhouette, len(Z)), replace=False)

    rows = []
    for k in k_list:
        km = KMeans(k, n_init=n_init, random_state=seed).fit(Z)
        lab = km.labels_
        sil = float(silhouette_score(Z[sub], lab[sub]))
        ch = float(calinski_harabasz_score(Z, lab))
        db = float(davies_bouldin_score(Z, lab))
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": sil, "calinski": ch, "davies_bouldin": db})

    # elbow on inertia (max distance to the chord joining first and last point)
    ks = np.array([r["k"] for r in rows], float)
    ine = np.array([r["inertia"] for r in rows], float)
    x = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    yv = (ine - ine.min()) / (ine.max() - ine.min() + 1e-12)
    # distance from each point to the line through (0,1)-(1,0)
    elbow_k = int(ks[int(np.argmax(np.abs(x + yv - 1.0) / np.sqrt(2)))])

    best_sil = int(max(rows, key=lambda r: r["silhouette"])["k"])
    best_ch = int(max(rows, key=lambda r: r["calinski"])["k"])
    best_db = int(min(rows, key=lambda r: r["davies_bouldin"])["k"])

    if verbose:
        print("=" * 74)
        print("LABEL-FREE SELECTION OF k   (no labels used anywhere below)")
        print("=" * 74)
        print(f"{'k':>4s} {'inertia':>12s} {'silhouette':>11s} "
              f"{'calinski':>10s} {'davies-b':>9s}")
        print("-" * 74)
        for r in rows:
            print(f"{r['k']:>4d} {r['inertia']:>12.0f} {r['silhouette']:>11.4f} "
                  f"{r['calinski']:>10.0f} {r['davies_bouldin']:>9.3f}")
        print("-" * 74)
        print(f"  silhouette (max)      -> k = {best_sil}")
        print(f"  Calinski-Harabasz(max)-> k = {best_ch}")
        print(f"  Davies-Bouldin (min)  -> k = {best_db}")
        print(f"  inertia elbow         -> k = {elbow_k}")
        votes = [best_sil, best_ch, best_db, elbow_k]
        vals, cnts = np.unique(votes, return_counts=True)
        consensus = int(vals[np.argmax(cnts)]) if cnts.max() > 1 else int(np.median(votes))
        print(f"  --> CONSENSUS k = {consensus}")
        print("=" * 74)

    votes = [best_sil, best_ch, best_db, elbow_k]
    vals, cnts = np.unique(votes, return_counts=True)
    consensus = int(vals[np.argmax(cnts)]) if cnts.max() > 1 else int(np.median(votes))
    return {"rows": rows, "silhouette_k": best_sil, "calinski_k": best_ch,
            "davies_bouldin_k": best_db, "elbow_k": elbow_k,
            "consensus_k": consensus}


# ---------------------------------------------------------------------------
# 2. STABILITY AT A GIVEN k
# ---------------------------------------------------------------------------
def stability(X, y_cls, y_bin, healthy_idx, k,
              seeds=range(10), n_init=500, verbose=True):
    """Run the k-label pipeline at fixed k over many seeds. Report mean +/- std."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    accs, f1s, aucs, cls_accs = [], [], [], []
    for s in seeds:
        Z = _pca(X, seed=0)                    # PCA is deterministic; fix it
        km = KMeans(k, n_init=n_init, random_state=s).fit(Z)
        lab, cen = km.labels_, km.cluster_centers_

        # name each cluster with ONE expert label (image nearest the centroid)
        names = {}
        for c in range(k):
            idx = np.where(lab == c)[0]
            if len(idx) == 0:
                continue
            d = np.linalg.norm(Z[idx] - cen[c], axis=1)
            names[c] = int(y_cls[idx[int(np.argmin(d))]])

        pred_cls = np.array([names.get(int(c), -1) for c in lab])
        pred_bin = (pred_cls != healthy_idx).astype(int)

        cls_accs.append(float((pred_cls == y_cls).mean()))
        accs.append(accuracy_score(y_bin, pred_bin))
        f1s.append(f1_score(y_bin, pred_bin, zero_division=0))

        h = [c for c, v in names.items() if v == healthy_idx]
        d_ = [c for c, v in names.items() if v != healthy_idx]
        if h and d_:
            D = np.stack([np.linalg.norm(Z - cen[c], axis=1) for c in range(k)])
            margin = D[h].min(axis=0) - D[d_].min(axis=0)
            aucs.append(roc_auc_score(y_bin, margin))
        else:
            aucs.append(np.nan)

    out = {"k": k, "n_seeds": len(list(seeds)), "n_init": n_init,
           "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
           "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
           "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
           "cls_acc_mean": float(np.mean(cls_accs)), "cls_acc_std": float(np.std(cls_accs)),
           "aucs": [float(a) for a in aucs]}
    if verbose:
        print(f"\n  k={k}, n_init={n_init}, {len(list(seeds))} seeds")
        print(f"    binary AUC : {out['auc_mean']:.4f} +/- {out['auc_std']:.4f}")
        print(f"    binary acc : {out['acc_mean']:.4f} +/- {out['acc_std']:.4f}")
        print(f"    binary F1  : {out['f1_mean']:.4f} +/- {out['f1_std']:.4f}")
        print(f"    7-class acc: {out['cls_acc_mean']:.4f} +/- {out['cls_acc_std']:.4f}")
        print(f"    range      : [{min(out['aucs']):.4f}, {max(out['aucs']):.4f}]")
    return out
