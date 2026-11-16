"""
multiclass_cluster.py -- can we get FINE-GRAINED disease classes without labels?

The binary pipeline uses a scalar attention-concentration score, which by
construction cannot say WHICH disease is present. But the frozen DINO feature
vector can: a supervised linear probe on these features reaches ~95% on 7
classes, so the class information is there. The question is whether it is
recoverable WITHOUT labels.

Method
  1. Extract frozen DINO [CLS] features for every image (no labels).
  2. Cluster into k=7 groups (k-means / agglomerative).
  3. Map clusters -> class names.

On step 3, be precise about what it costs:
  - HUNGARIAN matching (optimal 1-1 assignment maximising accuracy) uses the
    full label set. It is the standard metric in unsupervised-classification
    papers (DeepCluster, SCAN) and answers "how good is the CLUSTERING?" It is
    an EVALUATION device, not a deployable procedure.
  - ONE-SHOT naming: an expert looks at the single image closest to each cluster
    centroid and names it. Costs k labels (7), not 13,084. This IS deployable,
    and we report it separately because it is the honest practical number.

We report both, plus label-free cluster-quality metrics (NMI, ARI) that need no
mapping at all.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_accuracy(y_true, y_pred, n_classes=None):
    """
    Optimal 1-1 cluster->class assignment, then accuracy.
    Standard metric for unsupervised classification. Uses all labels =>
    an upper bound on what the clustering could deliver, not a deployable number.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    k = n_classes or max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[p, t] += 1
    row, col = linear_sum_assignment(-w)
    mapping = {int(r): int(c) for r, c in zip(row, col)}
    acc = w[row, col].sum() / len(y_true)
    return float(acc), mapping


def oneshot_naming_accuracy(y_true, y_pred, feats, centroids):
    """
    DEPLOYABLE mapping: for each cluster, take the single image nearest the
    centroid, reveal ONLY that image's label (k labels total), and name the
    whole cluster with it. Then score.

    Note this can leave a class unnamed and another duplicated -- that is a real
    property of the procedure, not a bug, and we report it.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mapping = {}
    for c in range(len(centroids)):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(feats[idx] - centroids[c], axis=1)
        rep = idx[int(np.argmin(d))]      # the ONE image the expert looks at
        mapping[c] = int(y_true[rep])     # the ONE label consumed
    pred_named = np.array([mapping.get(int(c), -1) for c in y_pred])
    acc = float((pred_named == y_true).mean())
    n_covered = len(set(mapping.values()))
    return acc, mapping, n_covered


def purity(y_true, y_pred):
    """Fraction of images in the majority class of their own cluster."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tot = 0
    for c in np.unique(y_pred):
        m = y_pred == c
        if m.sum():
            tot += np.bincount(y_true[m]).max()
    return float(tot / len(y_true))


def report(y_true, y_pred, feats, centroids, class_names, n_classes=7):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    h_acc, h_map = hungarian_accuracy(y_true, y_pred, n_classes)
    o_acc, o_map, n_cov = oneshot_naming_accuracy(y_true, y_pred, feats, centroids)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    pur = purity(y_true, y_pred)
    chance = 1.0 / n_classes

    print("=" * 70)
    print("UNSUPERVISED MULTI-CLASS DISEASE CLUSTERING")
    print("=" * 70)
    print(f"  chance (1/{n_classes})                : {chance*100:5.1f}%")
    print(f"  Hungarian accuracy (all labels)  : {h_acc*100:5.1f}%   [eval upper bound]")
    print(f"  One-shot naming ({n_classes} labels only) : {o_acc*100:5.1f}%   [DEPLOYABLE]")
    print(f"  distinct classes named           : {n_cov}/{n_classes}")
    print(f"  cluster purity                   : {pur*100:5.1f}%")
    print(f"  NMI (label-free)                 : {nmi:.3f}")
    print(f"  ARI (label-free)                 : {ari:.3f}")
    print("-" * 70)

    # confusion: which classes get merged?
    print("Cluster composition (rows = cluster, cols = true class):")
    k = len(centroids)
    w = np.zeros((k, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        w[p, t] += 1
    hdr = "       " + " ".join(f"{c[:6]:>7s}" for c in class_names)
    print(hdr)
    for c in range(k):
        dom = class_names[int(np.argmax(w[c]))] if w[c].sum() else "-"
        row = f"  c{c}: " + " ".join(f"{v:>7d}" for v in w[c])
        print(f"{row}   -> {dom}")
    print("=" * 70)

    return {"hungarian_acc": h_acc, "oneshot_acc": o_acc, "classes_named": n_cov,
            "purity": pur, "nmi": float(nmi), "ari": float(ari), "chance": chance}
