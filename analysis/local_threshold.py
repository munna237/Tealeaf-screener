#!/usr/bin/env python3
"""Triage threshold placed on a small LOCAL calibration set instead of the test split (Reviewer 2, minor 3).

A site labels n diseased leaves; the threshold is the k-th lowest of their student scores, k = floor(0.05 (n + 1)),
which by order statistics targets 95% sensitivity with no test labels. Achieved sensitivity and specificity are
measured on every other test leaf (near-duplicate groups of the calibration leaves removed), over many random
calibration sets for each of the five published students.

    python analysis/local_threshold.py artifacts/      -> local_threshold.json"""
import json, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ART = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
E = np.load(os.path.join(ART, 'E_scores_5rep.npz')); S, d = E['probe_distil'], E['dte']
X = np.load(os.path.join(ART, 'features_dinov1_crop.npz'))['Xte'].astype(float); X /= np.linalg.norm(X, axis=1, keepdims=True)
grp = connected_components(csr_matrix((X @ X.T) >= 0.95), directed=False)[1]
DRAWS, PREV, rng = 2000, 0.10, np.random.default_rng(2026)
pos = np.where(d == 1)[0]
cleared = lambda sens, spec: (1 - PREV) * spec + PREV * (1 - sens)      # fraction of leaves NOT sent to the expert at 10% prevalence

out = {'rule': 'threshold = k-th lowest score of n labelled diseased leaves, k = floor(0.05 (n + 1)); evaluated on the rest',
       'draws_per_student': DRAWS, 'students': int(S.shape[0])}
# reference: threshold placed on the test split itself (the optimistic placement now in the paper)
ref = []
for s in S:
    t = np.sort(s[d == 1])[int(np.floor(0.05 * (d == 1).sum()))]; f = s >= t
    ref.append((f[d == 1].mean(), (~f)[d == 0].mean()))
ref = np.array(ref)
out['test_split_placement'] = dict(sens=float(ref[:, 0].mean()), spec=[float(ref[:, 1].mean()), float(ref[:, 1].std(ddof=1))],
                                   cleared_10pct=float(np.mean([cleared(a, b) for a, b in ref])))
for n in (10, 20, 40, 80):
    k = max(1, int(np.floor(0.05 * (n + 1))))
    sens, spec, clr = [], [], []
    for s in S:
        for _ in range(DRAWS):
            cal = rng.choice(pos, n, replace=False)
            t = np.sort(s[cal])[k - 1]
            ev = ~np.isin(grp, grp[cal])
            f = s[ev] >= t; de = d[ev]
            a, b = f[de == 1].mean(), (~f)[de == 0].mean()
            sens.append(a); spec.append(b); clr.append(cleared(a, b))
    sens, spec, clr = map(np.array, (sens, spec, clr))
    out[f'n{n}'] = dict(labels=n, expert_minutes=round(n * 8 / 60, 1), k=k,
                        sens_mean=float(sens.mean()), sens_p05=float(np.percentile(sens, 5)), sens_p50=float(np.median(sens)),
                        p_sens_ge_090=float((sens >= 0.90).mean()), spec_mean=float(spec.mean()), spec_p05=float(np.percentile(spec, 5)),
                        cleared_10pct_mean=float(clr.mean()))
json.dump(out, open('local_threshold.json', 'w'), indent=1)
r = out['test_split_placement']
print('test-split placement (paper now): sens %.3f | spec %.3f ± %.3f | cleared at 10%% %.3f' % (r['sens'], *r['spec'], r['cleared_10pct']))
for n in (10, 20, 40, 80):
    v = out[f'n{n}']
    print('n=%2d diseased (%.1f min, k=%d): sens mean %.3f, median %.3f, 5th pct %.3f, P(>=0.90) %.2f | spec %.3f (5th pct %.3f) | cleared %.3f'
          % (n, v['expert_minutes'], v['k'], v['sens_mean'], v['sens_p50'], v['sens_p05'], v['p_sens_ge_090'], v['spec_mean'], v['spec_p05'], v['cleared_10pct_mean']))
