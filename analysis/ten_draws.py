"""Regenerate the ten-draw / ten-seed numbers and the Segment Anything diagnostic from released artifacts (CPU, ~1 min).

    python analysis/ten_draws.py <artifacts_dir>      -> ten_draws.json in the current directory

Covers: Table 12 and Section IV-M (Experiment E, ten label draws, including the frozen probe re-fitted from cached
features), Table 17 (field, ten seeds), Table 18 (blind transfer, ten seeds) and Table 6 (SAM diagnostic, from the
per-image file saved by notebooks/sam_confound_followup.ipynb)."""
import json, os, sys
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ART = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
J = lambda f: json.load(open(os.path.join(ART, f)))
out = {}

# ---- Experiment E, ten draws ----------------------------------------------------------------
a = np.load(os.path.join(ART, 'E_scores_5rep.npz')); b = np.load(os.path.join(ART, 'E_scores_rep5_9.npz'))
d = a['dte']
def metrics(s):
    p = s >= 0.5
    return dict(auc=roc_auc_score(d, s), acc=float((p == d).mean()), sens=float(p[d == 1].mean()), spec=float((~p)[d == 0].mean()))
E = {k: [metrics(a[k][r]) for r in range(5)] + [metrics(b[f'{k}_rep{r}']) for r in range(5, 10)] for k in ('labels_only', 'probe_distil')}
F = np.load(os.path.join(ART, 'features_dinov1_crop.npz')); Xtr, ytr, Xte, dte = F['Xtr'], F['ytr'], F['Xte'], F['dte']
assert (dte == d).all()
probe = []
for rep in range(10):
    r = np.random.default_rng(42 * 1000 + rep)
    i = np.concatenate([r.choice(np.where(ytr == c)[0], 10, replace=False) for c in range(7)])
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[i], ytr[i]); probe.append(roc_auc_score(dte, 1 - m.predict_proba(Xte)[:, 3]))
L = np.array([x['auc'] for x in E['labels_only']]); D = np.array([x['auc'] for x in E['probe_distil']]); P = np.array(probe)
summ = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v)))
out['E'] = dict(labels_only=summ(L), probe_distil=summ(D), probe=summ(P), probe_first5=summ(P[:5]),
                distil_acc=float(np.mean([x['acc'] for x in E['probe_distil']])), distil_sens=float(np.mean([x['sens'] for x in E['probe_distil']])),
                distil_spec=float(np.mean([x['spec'] for x in E['probe_distil']])),
                paired_t_p=float(stats.ttest_rel(D, L).pvalue), wilcoxon_p=float(stats.wilcoxon(D, L).pvalue),
                pitman_morgan_p=float(stats.pearsonr(D + L, D - L).pvalue), sd_ratio=float(L.std(ddof=1) / D.std(ddof=1)),
                spearman_gain_vs_labels_only=float(stats.spearmanr(D - L, L).statistic), spearman_student_vs_probe=float(stats.spearmanr(D, P).statistic),
                student_minus_probe_mean=float(np.mean(D - P)), student_minus_probe_min=float(np.min(D - P)))

# ---- Experiment F (field), ten seeds -------------------------------------------------------------
R = {**J('experiment_F_robustness.json'), **J('experiment_F_seeds3_9.json')}
f = {}
for S in ('S0', 'S1', 'S2'):
    rr = [R[f'{S}_rep{i}'] for i in range(10)]
    f[S] = dict(whole=summ([x['field_whole']['auc'] for x in rr]), centre=summ([x['field_centre']['auc'] for x in rr]),
                tealeaf=summ([x['tealeaf_test']['auc'] for x in rr]),
                spec_centre_fixed=summ([x['field_centre']['sens_spec_fixed_tealeaf_thr'][1] for x in rr]),
                sens_spec_whole_fixed=np.mean([x['field_whole']['sens_spec_fixed_tealeaf_thr'] for x in rr], 0).tolist())
g = np.array([R[f'S1_rep{i}']['field_whole']['auc'] - R[f'S0_rep{i}']['field_whole']['auc'] for i in range(10)])
gc = np.array([R[f'S1_rep{i}']['field_centre']['auc'] - R[f'S0_rep{i}']['field_centre']['auc'] for i in range(10)])
f['S1_minus_S0'] = dict(whole_min=float(g.min()), whole_max=float(g.max()), centre_min=float(gc.min()), centre_max=float(gc.max()),
                        wins=int((g > 0).sum()), wilcoxon_p=float(stats.wilcoxon(g).pvalue))
out['F'] = f

# ---- blind transfer, ten seeds -----------------------------------------------------------------
bt = J('external_blind_transfer_10seeds.json')['per_model']; bo = {}
for e in ('External_1', 'External_2'):
    v = {S: np.array([bt[f'{e}/{S}/rep{i}']['auc'] for i in range(10)]) for S in ('S0', 'S1', 'S2')}
    gg = v['S1'] - v['S0']
    bo[e] = {**{S: summ(v[S]) for S in v}, 'wins': int((gg > 0).sum()), 'median_gain': float(np.median(gg)), 'wilcoxon_p': float(stats.wilcoxon(gg).pvalue)}
out['blind'] = bo

# ---- Segment Anything diagnostic (Table 6) -----------------------------------------------------------
Z = np.load(os.path.join(ART, 'sam_confound_per_image.npz'), allow_pickle=True)
DS = sorted({k.split('|')[0] for k in Z.files})
CV = StratifiedKFold(5, shuffle=True, random_state=0)
dir_ = lambda x: max(x, 1 - x)
def shape_auc(S, Y):
    return float(max(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), S, Y, cv=CV, scoring='roc_auc').mean(),
                     cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), S, Y, cv=CV, scoring='roc_auc').mean()))
sam = {}
for name in DS:
    g_ = lambda k: Z[f'{name}|{k}']
    Y, A, S, G, W = g_('y'), g_('area'), g_('shape'), g_('G'), g_('W'); ok = ~np.isnan(W)
    row = dict(n=int(len(Y)), area_ratio_mean=float(A[Y == 0].mean() / A[Y == 1].mean()), area_ratio_median=float(np.median(A[Y == 0]) / np.median(A[Y == 1])),
               shape_only_auc=shape_auc(S, Y), collapse=float(dir_(roc_auc_score(Y, G)) - dir_(roc_auc_score(Y[ok], W[ok]))))
    if name == 'TeaLeafDiseaseBD':
        k = (g_('iou_pred') >= 0.88) & (g_('stab') >= 0.95) & ~g_('border')
        row['qc_border_area_ratio_mean'] = float(A[k & (Y == 0)].mean() / A[k & (Y == 1)].mean())
        io = g_('iou_otsu'); row['frac_iou_below_0_5'] = float(np.nanmean(io < 0.5)); row['median_iou'] = float(np.nanmedian(io))
    sam[name] = row
out['sam'] = sam
json.dump(out, open('ten_draws.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('E',)}, indent=1)[:1500])
for n, r in sam.items(): print('%-18s ratio %.2f (median %.2f) shape %.3f collapse %+.3f' % (n, r['area_ratio_mean'], r['area_ratio_median'], r['shape_only_auc'], r['collapse']))
