"""Follow-ups for the external test (CPU only, cached DINO features).
A. reproduce fixed-threshold zero-shot numbers (check against external_results.json)
B. label-free threshold: tau = quantile of external scores at the healthy prevalence pi
   (same rule as the paper's pseudo-labelling); also pi misestimated by +/-10 points
C. local recalibration: n labelled leaves per TARGET class (1,2,5,10), refit, test on the rest
   C1 target-only multinomial probe; C2 pooled binary probe (source 70 labels + target labels)
Outputs: recalibration_results.json (+ printed tables)."""
import numpy as np, json, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
D = sys.argv[1] if len(sys.argv) > 1 else '.'
F = np.load(f'{D}/features_dinov1_crop.npz'); Xtr, ytr, Xte, dte = F['Xtr'], F['ytr'], F['Xte'], F['dte']
J = json.load(open(f'{D}/E_distillation_5rep.json')); EXT = json.load(open(f'{D}/external_results.json'))
SEED, R, H, TS = 42, 5, 3, 0.95
DUP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
EXTS = {'EWU': np.load(f'{D}/feat_EWU_box.npz'), 'Allynugger': np.load(f'{D}/feat_Ally_whole.npz')}
ms = lambda a: [float(np.mean(a)), float(np.std(a, ddof=1))]

def draw_src(rep):
    r = np.random.default_rng(SEED * 1000 + rep)
    return np.concatenate([r.choice(np.where(ytr == c)[0], 10, replace=False) for c in range(7)])
P = [LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[draw_src(r)], ytr[draw_src(r)]) for r in range(R)]
assert np.allclose([roc_auc_score(dte, 1 - p.predict_proba(Xte)[:, H]) for p in P], J['probe_auc_per_rep'], atol=1e-4)
def thr95(s, d):
    pos = np.sort(s[d == 1]); return pos[int(np.floor((1 - TS) * len(pos)))]
THR = [thr95(1 - p.predict_proba(Xte)[:, H], dte) for p in P]
def ss(s, y, t):
    f = s >= t; return f[y == 1].mean(), (~f)[y == 0].mean()
def spec_at_sens(s, y):                       # oracle operating point (uses test labels; for reference)
    return ss(s, y, thr95(s, y))[1]

out = {}
for name, E in EXTS.items():
    X, y, cls = E['X'], E['y'], E['cls']; pi = float((y == 0).mean()); res = {'n': int(len(y)), 'healthy_frac': pi}
    S0 = [1 - p.predict_proba(X)[:, H] for p in P]
    # A. zero-shot, fixed in-distribution threshold
    a = [roc_auc_score(y, s) for s in S0]; st = [ss(s, y, t) for s, t in zip(S0, THR)]
    key = 'EWU_box_probe' if name == 'EWU' else 'Ally_whole_probe'
    assert abs(np.mean(a) - EXT[key]['auc'][0]) < 1e-3, 'zero-shot AUC did not reproduce'
    res['zero_shot_fixed'] = dict(auc=ms(a), sens=ms([x[0] for x in st]), spec=ms([x[1] for x in st]),
                                  oracle_spec_at_95sens=ms([spec_at_sens(s, y) for s in S0]))
    # B. label-free prevalence-quantile threshold
    lf = {}
    for tag, p_ in (('true_pi', pi), ('pi_minus_10', pi - .10), ('pi_plus_10', pi + .10)):
        st = [ss(s, y, np.quantile(s, p_)) for s in S0]
        lf[tag] = dict(pi_used=float(p_), sens=ms([x[0] for x in st]), spec=ms([x[1] for x in st]))
    res['label_free_threshold'] = lf
    # C. local recalibration
    # near-duplicate clusters: connected components at cosine >= DUP
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    A = (X @ X.T) >= DUP; grp = connected_components(csr_matrix(A), directed=False)[1]
    res['dup_clusters'] = dict(threshold=DUP, n_clusters=int(grp.max() + 1),
                               largest=int(np.bincount(grp).max()), singletons=int((np.bincount(grp) == 1).sum()))
    classes = np.unique(cls); K = len(classes); cid = np.searchsorted(classes, cls); hk = int(np.where(classes == 'Healthy')[0][0])
    rc = {}
    for n in (1, 2, 5, 10):
        r1, r2 = {'auc': [], 'spec95': []}, {'auc': [], 'spec95': []}
        for rep in range(R):
            g = np.random.default_rng(7000 + 100 * n + rep)
            lab = np.concatenate([g.choice(np.where(cid == k)[0], n, replace=False) for k in range(K)])
            # group-aware: drop every image in the same near-duplicate cluster as a labelled one
            te = np.where(~np.isin(grp, grp[lab]))[0]
            if True:
                m1 = LogisticRegression(C=1.0, max_iter=2000).fit(X[lab], cid[lab])
                s1 = 1 - m1.predict_proba(X[te])[:, list(m1.classes_).index(hk)]
                r1['auc'].append(roc_auc_score(y[te], s1)); r1['spec95'].append(spec_at_sens(s1, y[te]))
            si = draw_src(rep)
            Xp = np.concatenate([Xtr[si], X[lab]]); yp = np.concatenate([(ytr[si] != H).astype(int), y[lab]])
            m2 = LogisticRegression(C=1.0, max_iter=2000).fit(Xp, yp); s2 = m2.predict_proba(X[te])[:, 1]
            r2['auc'].append(roc_auc_score(y[te], s2)); r2['spec95'].append(spec_at_sens(s2, y[te]))
        rc[n] = {'labels': int(n * K), 'expert_min': n * K * 8 / 60,
                 'target_only': {k: ms(v) for k, v in r1.items()}, 'pooled_source_plus_target': {k: ms(v) for k, v in r2.items()}}
    res['local_recalibration'] = rc
    out[name] = res
json.dump(out, open('recalibration_results.json', 'w'), indent=2)

f = lambda v: '%.3f ± %.3f' % tuple(v)
for name, r in out.items():
    z = r['zero_shot_fixed']; print(f'\n=== {name}  n={r["n"]}  healthy={r["healthy_frac"]:.3f}')
    print('zero-shot fixed thr : AUC', f(z['auc']), '| sens', f(z['sens']), '| spec', f(z['spec']), '| oracle spec@95', f(z['oracle_spec_at_95sens']))
    for k, v in r['label_free_threshold'].items():
        print('label-free %-11s (pi=%.2f): sens %s | spec %s' % (k, v['pi_used'], f(v['sens']), f(v['spec'])))
    for n, v in r['local_recalibration'].items():
        print('n=%2d/class (%3d labels, %.1f min): target-only AUC %s spec@95 %s | pooled AUC %s spec@95 %s' % (
            n, v['labels'], v['expert_min'], f(v['target_only']['auc']), f(v['target_only']['spec95']),
            f(v['pooled_source_plus_target']['auc']), f(v['pooled_source_plus_target']['spec95'])))
