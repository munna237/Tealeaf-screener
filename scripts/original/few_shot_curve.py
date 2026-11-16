"""
few_shot_curve.py -- how few expert labels does a tea garden actually need?

THE AGRICULTURAL QUESTION
Not "can we use zero labels" (we showed that fails: the attention score is a
leaf-scale artefact, and label-free clustering does not beat leaf geometry).
The question an extension service actually asks is:

    "If I hire an agronomist for one afternoon instead of one month,
     what accuracy do I get?"

METHOD
Frozen DINO features of LEAF-CROP-NORMALISED images (so camera distance cannot
drive the result -- this matters in the field). Train a linear probe on n labelled
images PER CLASS. Sweep n = 1, 2, 5, 10, 20, 50, 100, all.

CONTROLS (both essential)
  - leaf-shape features under the identical protocol. If shape does as well,
    the model is reading morphology, not pathology.
  - a random-guess floor.

We report BOTH tasks a garden cares about:
  binary  : healthy vs diseased        (screening: what needs a human look?)
  7-class : which disease              (what treatment do I apply?)

Every point is mean +/- std over repeated draws of the labelled seed set, because
WHICH leaves the expert happens to label matters when there are only a few.
"""

import numpy as np


def _probe(Xtr, ytr, Xte, yte, multi):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    # NB: newer scikit-learn removed `multi_class`; multinomial is inferred.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    if multi:
        return {"acc": acc,
                "f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                "auc": float("nan")}
    prob = clf.predict_proba(Xte)[:, 1]
    return {"acc": acc,
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "auc": float(roc_auc_score(yte, prob))}


def curve(X, y_cls, y_bin, class_names, healthy_idx,
          shots=(1, 2, 5, 10, 20, 50, 100), n_repeats=10,
          test_frac=0.3, seed=0, shape_feats=None, verbose=True):
    """
    Returns a dict keyed by n-shot with binary and 7-class performance.
    The labelled seed set is drawn ONLY from the train split; the test split is
    never touched by the expert.
    """
    rng = np.random.default_rng(seed)
    K = len(class_names)
    n = len(y_bin)

    res = {"shots": [], "binary": [], "multiclass": [],
           "shape_binary": [], "labels_used": []}

    for s in shots:
        b_auc, b_acc, b_f1, m_acc, m_f1, sh_auc = [], [], [], [], [], []
        for rep in range(n_repeats):
            r = np.random.default_rng(seed * 1000 + rep)
            # fixed test split per repeat
            perm = r.permutation(n)
            n_te = int(test_frac * n)
            te, tr_pool = perm[:n_te], perm[n_te:]

            # the expert labels s images PER CLASS, drawn from the train pool
            picked = []
            ok = True
            for c in range(K):
                idx = tr_pool[y_cls[tr_pool] == c]
                if len(idx) < s:
                    ok = False; break
                picked.append(r.choice(idx, s, replace=False))
            if not ok:
                continue
            tr = np.concatenate(picked)

            m = _probe(X[tr], y_cls[tr], X[te], y_cls[te], multi=True)
            m_acc.append(m["acc"]); m_f1.append(m["f1"])

            b = _probe(X[tr], y_bin[tr], X[te], y_bin[te], multi=False)
            b_auc.append(b["auc"]); b_acc.append(b["acc"]); b_f1.append(b["f1"])

            if shape_feats is not None:
                sb = _probe(shape_feats[tr], y_bin[tr], shape_feats[te],
                            y_bin[te], multi=False)
                sh_auc.append(sb["auc"])

        if not b_auc:
            continue
        res["shots"].append(s)
        res["labels_used"].append(s * K)
        res["binary"].append({
            "auc": float(np.mean(b_auc)), "auc_std": float(np.std(b_auc)),
            "acc": float(np.mean(b_acc)), "f1": float(np.mean(b_f1))})
        res["multiclass"].append({
            "acc": float(np.mean(m_acc)), "acc_std": float(np.std(m_acc)),
            "f1": float(np.mean(m_f1))})
        res["shape_binary"].append(
            {"auc": float(np.mean(sh_auc)) if sh_auc else float("nan")})

    if verbose:
        print("=" * 82)
        print("LABEL EFFICIENCY: how many leaves must an expert label?")
        print(f"  {len(y_bin):,} images | {K} classes | "
              f"test = {int(test_frac*100)}% held out | {n_repeats} repeats")
        print("=" * 82)
        print(f"{'per class':>9s} {'TOTAL':>7s} {'expert':>8s} | "
              f"{'BINARY auc':>12s} {'acc':>7s} | {'7-CLASS acc':>12s} | "
              f"{'shape ctrl':>10s}")
        print(f"{'':>9s} {'labels':>7s} {'minutes':>8s} |")
        print("-" * 82)
        for i, s in enumerate(res["shots"]):
            tot = res["labels_used"][i]
            mins = tot * 8 / 60.0          # 8 s per label
            b = res["binary"][i]; m = res["multiclass"][i]
            sc = res["shape_binary"][i]["auc"]
            print(f"{s:>9d} {tot:>7d} {mins:>8.1f} | "
                  f"{b['auc']:>7.4f}+/-{b['auc_std']:.3f} {b['acc']:>7.3f} | "
                  f"{m['acc']:>7.3f}+/-{m['acc_std']:.3f} | {sc:>10.4f}")
        print("-" * 82)
        print("  'expert minutes' assumes 8 s to label one leaf.")
        print("  Full supervision would require "
              f"{len(y_bin):,} labels = {len(y_bin)*8/3600:.1f} expert-hours.")
        print("=" * 82)

    return res
