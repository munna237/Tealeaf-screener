"""CPU-only reconciliation of TeaLeafDiseaseBD paper numbers from ONE run (revision_v1, E 5-rep).
Inputs : E_scores_5rep.npz, features_dinov1_crop.npz, _split.json, E_distillation_5rep.json
Outputs: reconciled_numbers.json, fig_confusion.png, fig_roc_pr.png, printed LaTeX-ready rows."""
import numpy as np, json, re, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix, accuracy_score)
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
D = sys.argv[1] if len(sys.argv) > 1 else '.'
E = np.load(f'{D}/E_scores_5rep.npz'); F = np.load(f'{D}/features_dinov1_crop.npz')
SPL = json.load(open(f'{D}/_split.json')); J = json.load(open(f'{D}/E_distillation_5rep.json'))
Xtr, ytr, Xte, yte, dte = F['Xtr'], F['ytr'], F['Xte'], F['yte'], F['dte']
assert (E['dte'] == dte).all() and len(dte) == SPL['n_test'] == J['n_test']
CLASSES = ['Algal Leaf Spot','Brown Blight','Gray Blight','Healthy','Helopeltis','Red Rust','Red Spider']
H = 3; SEED = J['_cfg']['SEED']; R = J['n_reps']
S = E['probe_distil']; L = E['labels_only']
out = {'source': 'revision_v1 E_distillation_5rep (prevalence-quantile pseudo-labels), test n=%d' % len(dte)}
sd = lambda a: float(np.std(a, ddof=1))
def ms(a): return {'mean': float(np.mean(a)), 'sd': sd(a), 'per_rep': [float(x) for x in a]}

# ---- 1. sanity: AUCs reproduce the JSON
auc_s = np.array([roc_auc_score(dte, s) for s in S]); auc_l = np.array([roc_auc_score(dte, s) for s in L])
assert np.allclose(auc_s, J['per_rep']['probe_distil'], atol=1e-6)
assert np.allclose(auc_l, J['per_rep']['labels_only'], atol=1e-6)

# ---- 2. deployed student operating metrics at threshold 0.5
def op(d, s, t=0.5):
    p = (s >= t).astype(int); tn, fp, fn, tp = confusion_matrix(d, p, labels=[0, 1]).ravel()
    return dict(acc=(tp+tn)/len(d), sens=tp/(tp+fn), spec=tn/(tn+fp), bal_acc=0.5*(tp/(tp+fn)+tn/(tn+fp)))
ops = [op(dte, s) for s in S]
out['student'] = {'auc': ms(auc_s), **{k: ms([o[k] for o in ops]) for k in ops[0]}}
out['labels_only'] = {'auc': ms(auc_l), **{k: ms([op(dte, s)[k] for s in L]) for k in ops[0]}}
med = int(np.argmin(np.abs(auc_s - auc_s.mean())))
out['median_seed'] = {'rep': med, 'auc': float(auc_s[med])}

# ---- 3. probe (teacher) reproduced from cached features, same seed draws, C=1.0
def draw(rep, n):
    rng = np.random.default_rng(SEED * 1000 + rep)
    return np.concatenate([rng.choice(np.where(ytr == c)[0], n, replace=False) for c in range(7)])
def probe(rep, n, C=1.0):
    idx = draw(rep, n); m = LogisticRegression(C=C, max_iter=2000).fit(Xtr[idx], ytr[idx])
    return m.predict_proba(Xte)
PR = [probe(r, 10) for r in range(R)]
pauc = np.array([roc_auc_score(dte, 1 - p[:, H]) for p in PR])
assert np.allclose(pauc, J['probe_auc_per_rep'], atol=1e-4), 'probe did not reproduce'
p7 = np.array([accuracy_score(yte, p.argmax(1)) for p in PR])
# binary decision = argmax class is not Healthy (P(diseased)>=0.5 is degenerate: 6 disease classes sum)
pbin_acc = np.array([accuracy_score(dte, (p.argmax(1) != H).astype(int)) for p in PR])
out['probe'] = {'auc': ms(pauc), 'binary_acc_argmax': ms(pbin_acc), 'acc_7class': ms(p7)}
cm7 = confusion_matrix(yte, PR[med].argmax(1), labels=range(7))
rec = cm7.diagonal() / cm7.sum(1)
out['probe_7class_recall_median_seed'] = {c: float(r) for c, r in zip(CLASSES, rec)}
rec_all = np.array([confusion_matrix(yte, p.argmax(1), labels=range(7)).diagonal() /
                    np.bincount(yte, minlength=7) for p in PR])
out['probe_7class_recall_5rep'] = {c: {'mean': float(rec_all[:, i].mean()), 'sd': sd(rec_all[:, i])}
                                   for i, c in enumerate(CLASSES)}
# sensitivity of 7-class accuracy to probe regularisation (explains 0.69 vs 0.75)
out['acc_7class_vs_C'] = {str(C): float(np.mean([accuracy_score(yte, probe(r, 10, C).argmax(1))
                                                  for r in range(R)])) for C in (1.0, 10.0, 100.0)}

# ---- 4. label-efficiency curve on the SAME features/split (C=1.0)
curve = {}
for n in (1, 2, 5, 10, 20, 50, 100):
    ps = [probe(r, n) for r in range(R)]
    a = [roc_auc_score(dte, 1 - p[:, H]) for p in ps]
    ba = [accuracy_score(dte, (p.argmax(1) != H).astype(int)) for p in ps]
    a7 = [accuracy_score(yte, p.argmax(1)) for p in ps]
    curve[n] = {'labels': 7 * n, 'expert_min': 7 * n * 8 / 60, 'auc': ms(a), 'bin_acc': ms(ba), 'acc7': ms(a7)}
out['label_curve_C1'] = curve

# ---- 5. triage: per-rep threshold hitting target sensitivity on test split
def at_sens(d, s, target):
    pos = np.sort(s[d == 1]); t = pos[int(np.floor((1 - target) * len(pos)))]  # >= t flagged
    flag = s >= t; sens = flag[d == 1].mean(); spec = (~flag)[d == 0].mean()
    cleared = (~flag).mean(); npv = ((~flag) & (d == 0)).sum() / max((~flag).sum(), 1)
    return dict(sens=sens, spec=spec, cleared=cleared, missed=1 - sens, npv=npv)
tri = {}
for tg in (0.90, 0.95, 0.98, 0.99):
    rs = [at_sens(dte, s, tg) for s in S]
    tri['test_%.2f' % tg] = {k: ms([r[k] for r in rs]) for k in rs[0]}
# field prevalence: same 95% threshold, prevalence-reweighted (spec, sens fixed by construction)
r95 = [at_sens(dte, s, 0.95) for s in S]
for prev in (0.30, 0.20, 0.10):
    cl = [(1 - prev) * r['spec'] + prev * (1 - r['sens']) for r in r95]
    npv = [(1 - prev) * r['spec'] / c for r, c in zip(r95, cl)]
    tri['field_%d%%' % round(prev * 100)] = {'sens': ms([r['sens'] for r in r95]),
        'spec': ms([r['spec'] for r in r95]), 'cleared': ms(cl), 'missed': ms([1 - r['sens'] for r in r95]), 'npv': ms(npv)}
out['triage'] = tri

# ---- 6. capture-session / device check (dates & camera prefix parsed from filenames)
P = SPL['test_paths']
date = np.array([re.search(r'_(\d{8})_', p).group(1) for p in P])
dev = np.array([p.split('/')[-1][:3] for p in P])
sess = {}
for name, arr in (('date', date), ('file_prefix', dev)):
    for v in np.unique(arr):
        m = arr == v; nh = int((dte[m] == 0).sum()); nd = int((dte[m] == 1).sum())
        e = {'n': int(m.sum()), 'healthy': nh, 'diseased': nd}
        if nh >= 20 and nd >= 20:
            e['student_auc'] = ms([roc_auc_score(dte[m], s[m]) for s in S])
        sess[f'{name}={v}'] = e
pri = {v: dte[date == v].mean() for v in np.unique(date)}
sess['date_prior_auc_in_sample'] = float(roc_auc_score(dte, [pri[v] for v in date]))
out['session_check'] = sess

# ---- 7. statistics from the JSON, re-derived
g = auc_s - auc_l
out['stats'] = {'paired_t': [float(x) for x in stats.ttest_rel(auc_s, auc_l)],
                'wilcoxon_p': float(stats.wilcoxon(auc_s, auc_l).pvalue),
                'sd_ratio': sd(auc_l) / sd(auc_s), 'spearman_gain_vs_labelsonly': float(stats.spearmanr(g, auc_l)[0]),
                'distil_minus_teacher': float(auc_s.mean() - pauc.mean())}
# Pitman-Morgan: corr(sum, diff) test for equal paired variances
u, v = auc_l + auc_s, auc_l - auc_s; r = np.corrcoef(u, v)[0, 1]; n = R
t = r * np.sqrt((n - 2) / (1 - r ** 2)); out['stats']['pitman_morgan_p'] = float(2 * stats.t.sf(abs(t), n - 2))

# ---- 8. bootstrap 95% CI on the median-seed student AUC (over test images)
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    i = rng.integers(0, len(dte), len(dte))
    if dte[i].min() != dte[i].max(): bs.append(roc_auc_score(dte[i], S[med][i]))
out['median_seed_auc_ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

json.dump(out, open('reconciled_numbers.json', 'w'), indent=2, default=float)

# ---- figures from the median seed
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.4]})
cmb = confusion_matrix(dte, (S[med] >= 0.5).astype(int), labels=[0, 1]); cmbn = cmb / cmb.sum(1, keepdims=True)
ax[0].imshow(cmbn, cmap='Blues', vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, '%.0f%%\n(%d)' % (100 * cmbn[i, j], cmb[i, j]), ha='center', va='center',
                   color='white' if cmbn[i, j] > .5 else 'black', fontsize=12)
ax[0].set_xticks([0, 1], ['Healthy', 'Diseased']); ax[0].set_yticks([0, 1], ['Healthy', 'Diseased'])
ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('True')
ax[0].set_title('Deployed student (binary)\nseed %d, AUC %.3f; 5-seed AUC %.3f ± %.3f' % (med, auc_s[med], auc_s.mean(), sd(auc_s)), fontsize=10)
c7n = cm7 / cm7.sum(1, keepdims=True); ax[1].imshow(c7n, cmap='Blues', vmin=0, vmax=1)
for i in range(7):
    for j in range(7):
        if c7n[i, j] >= .01: ax[1].text(j, i, '%d' % round(100 * c7n[i, j]), ha='center', va='center',
                                        color='white' if c7n[i, j] > .5 else 'black', fontsize=8)
ax[1].set_xticks(range(7), CLASSES, rotation=40, ha='right', fontsize=8); ax[1].set_yticks(range(7), CLASSES, fontsize=8)
ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('True')
ax[1].set_title('Frozen-feature probe, 7-class (70 labels), row-normalised %%\nseed %d, acc %.3f; 5-seed acc %.3f ± %.3f' % (med, p7[med], p7.mean(), sd(p7)), fontsize=10)
plt.tight_layout(); plt.savefig('fig_confusion.png', dpi=300)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
fpr, tpr, _ = roc_curve(dte, S[med]); ax[0].plot(fpr, tpr, lw=2, label='AUC = %.3f' % auc_s[med])
r = at_sens(dte, S[med], 0.95)
ax[0].plot(1 - r['spec'], r['sens'], '*', ms=14, color='C1', label='triage (sens %.0f%%, spec %.0f%%)' % (100 * r['sens'], 100 * r['spec']))
ax[0].plot([0, 1], [0, 1], ':', color='gray'); ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate')
ax[0].set_title('ROC: deployed student (seed %d)' % med); ax[0].legend(loc='lower right')
pr_, rc_, _ = precision_recall_curve(dte, S[med]); ap = average_precision_score(dte, S[med])
ax[1].plot(rc_, pr_, lw=2, label='AP = %.3f' % ap); ax[1].axhline(dte.mean(), ls=':', color='C1', label='prevalence (%.1f%%)' % (100 * dte.mean()))
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision–recall'); ax[1].legend(loc='lower left')
plt.tight_layout(); plt.savefig('fig_roc_pr.png', dpi=300)
print(json.dumps({k: out[k] for k in ('student', 'probe', 'median_seed', 'median_seed_auc_ci95', 'stats', 'acc_7class_vs_C')}, indent=1, default=float))
