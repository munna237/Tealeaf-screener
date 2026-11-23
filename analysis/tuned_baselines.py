"""Paired statistics for the tuned supervised baselines and ImageNet probes (Table 12, Table 2 note,
Section IV-M) and their field / external transfer (Tables 17 and 18, Section IV-Q). CPU, seconds.

    python analysis/tuned_baselines.py <artifacts_dir>   -> tuned_baselines.json in the current directory

Inputs: baseline_tuned.json (notebooks/baseline_tuned.ipynb), baseline_transfer.json
(notebooks/baseline_transfer.ipynb), experiment_F_robustness.json + experiment_F_seeds3_9.json (S0/S1),
external_blind_transfer_10seeds.json (S1 on the external datasets)."""
import json, os, sys
import numpy as np
from scipy import stats

ART = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
J = lambda f: json.load(open(os.path.join(ART, f)))
FT = ('ft_lr1e-4_strongaug', 'ft_lr3e-4_strongaug')
summ = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), min=float(np.min(v)), max=float(np.max(v)))
def paired(a, b):  # a - b over draws
    a, b = np.asarray(a), np.asarray(b); d = a - b
    return dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), t=float(stats.ttest_rel(a, b).statistic),
                t_p=float(stats.ttest_rel(a, b).pvalue), wilcoxon_p=float(stats.wilcoxon(a, b).pvalue))
pitman_morgan = lambda a, b: float(stats.pearsonr(np.add(a, b), np.subtract(a, b)).pvalue)

out = {}
B = J('baseline_tuned.json'); D = np.array(B['probe_distil']['per_draw'])
L = np.array(B['baselines']['labels_only_published']['per_draw'])
# DINO probe re-fitted from cached features (identical to analysis/ten_draws.py)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
F = np.load(os.path.join(ART, 'features_dinov1_crop.npz')); Xtr, ytr, Xte, dte = F['Xtr'], F['ytr'], F['Xte'], F['dte']
P = []
for rep in range(10):
    r = np.random.default_rng(42 * 1000 + rep)
    i = np.concatenate([r.choice(np.where(ytr == c)[0], 10, replace=False) for c in range(7)])
    P.append(roc_auc_score(dte, 1 - LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[i], ytr[i]).predict_proba(Xte)[:, 3]))
P = np.array(P)
t12 = {'probe_distil': summ(D), 'dino_probe': summ(P), 'labels_only_student_recipe': {**summ(L), **paired(D, L)}}
for k in FT:
    v = np.array(B['baselines'][k]['per_draw'])
    t12[k] = {**summ(v), 'distil_minus_ft': paired(D, v), 'pitman_morgan_p': pitman_morgan(D, v)}
for k in ('probe_imagenet_mobilenet_v3_small', 'probe_imagenet_resnet50'):
    v = np.array(B['baselines'][k]['per_draw'])
    t12[k] = {**summ(v), 'dino_minus_imagenet': paired(P, v)}
t12['bonferroni_alpha'] = 0.05 / 4
t12['all_four_added_survive'] = bool(all(t12[k]['dino_minus_imagenet']['t_p'] < 0.0125 and t12[k]['dino_minus_imagenet']['wilcoxon_p'] < 0.0125
                                         for k in ('probe_imagenet_mobilenet_v3_small', 'probe_imagenet_resnet50')))
out['table12'] = t12

T = J('baseline_transfer.json')['per_model']
R = {**J('experiment_F_robustness.json'), **J('experiment_F_seeds3_9.json')}
S = {s: {v: np.array([R[f'{s}_rep{i}'][f'field_{v}']['auc'] for i in range(10)]) for v in ('whole', 'centre')} for s in ('S0', 'S1')}
bt = J('external_blind_transfer_10seeds.json')['per_model']
t17 = {}
for k in FT:
    m = [T[f'{k}_rep{i}'] for i in range(10)]; row = {}
    for v in ('whole', 'centre'):
        a = np.array([x[f'field_{v}']['auc_heldout'] for x in m])
        row[v] = {**summ(a), 'all2500': float(np.mean([x[f'field_{v}']['auc_all2500'] for x in m])),
                  'sens_spec_fixed': np.mean([x[f'field_{v}']['sens_spec_fixed'] for x in m], 0).tolist(),
                  'S1_minus_ft': paired(S['S1'][v], a), 'ft_minus_S0': paired(a, S['S0'][v])}
    row['tealeaf'] = summ([x['tealeaf_auc'] for x in m])
    row['tealeaf_max_abs_diff_vs_table12'] = float(np.max(np.abs(np.array([x['tealeaf_auc'] for x in m]) - np.array(B['baselines'][k]['per_draw']))))
    for e in ('External_1', 'External_2'):
        a = np.array([x[e]['auc'] for x in m]); s1 = np.array([bt[f'{e}/S1/rep{i}']['auc'] for i in range(10)])
        row[e] = {**summ(a), 'sens_spec_fixed_mean': np.mean([x[e]['sens_spec_fixed'] for x in m], 0).tolist(),
                  'sens_spec_fixed_sd': np.std([x[e]['sens_spec_fixed'] for x in m], 0, ddof=1).tolist(), 'S1_minus_ft': paired(s1, a)}
    t17[k] = row
out['transfer'] = t17
json.dump(out, open('tuned_baselines.json', 'w'), indent=1)
f = lambda x: f"{x['mean']:.3f}+-{x['sd']:.3f} ({x['min']:.3f}-{x['max']:.3f})"
print('distil', f(t12['probe_distil']), '| DINO probe', f(t12['dino_probe']))
print('labels-only (student recipe)', f(t12['labels_only_student_recipe']), 't=%.2f p=%.3f W=%.4f' % (t12['labels_only_student_recipe']['t'], t12['labels_only_student_recipe']['t_p'], t12['labels_only_student_recipe']['wilcoxon_p']))
for k in FT: x = t12[k]; print(k, f(x), 'distil-ft %+.3f t_p=%.2f W=%.2f wins %d PM=%.2f' % (x['distil_minus_ft']['mean_diff'], x['distil_minus_ft']['t_p'], x['distil_minus_ft']['wilcoxon_p'], x['distil_minus_ft']['wins'], x['pitman_morgan_p']))
for k in ('probe_imagenet_mobilenet_v3_small', 'probe_imagenet_resnet50'): x = t12[k]; print(k, f(x), 'DINO-it %+.3f t_p=%.4f W=%.4f wins %d' % (x['dino_minus_imagenet']['mean_diff'], x['dino_minus_imagenet']['t_p'], x['dino_minus_imagenet']['wilcoxon_p'], x['dino_minus_imagenet']['wins']))
print('Bonferroni survive:', t12['all_four_added_survive'])
for k in FT:
    x = t17[k]; print(k, 'ours %.3f+-%.3f maxdiff %.4f' % (x['tealeaf']['mean'], x['tealeaf']['sd'], x['tealeaf_max_abs_diff_vs_table12']))
    for v in ('whole', 'centre'):
        y = x[v]; print('  field', v, '%.3f+-%.3f ss %.2f/%.2f all %.3f | S1-ft %+.3f wins %d W=%.4f t_p=%.5f | ft-S0 %+.3f wins %d W=%.3f t_p=%.3f' % (y['mean'], y['sd'], *y['sens_spec_fixed'], y['all2500'], y['S1_minus_ft']['mean_diff'], y['S1_minus_ft']['wins'], y['S1_minus_ft']['wilcoxon_p'], y['S1_minus_ft']['t_p'], y['ft_minus_S0']['mean_diff'], y['ft_minus_S0']['wins'], y['ft_minus_S0']['wilcoxon_p'], y['ft_minus_S0']['t_p']))
    for e in ('External_1', 'External_2'):
        y = x[e]; print('  ', e, '%.3f+-%.3f sens %.2f+-%.2f spec %.2f+-%.2f | S1-ft %+.3f t_p=%.2f W=%.2f' % (y['mean'], y['sd'], y['sens_spec_fixed_mean'][0], y['sens_spec_fixed_sd'][0], y['sens_spec_fixed_mean'][1], y['sens_spec_fixed_sd'][1], y['S1_minus_ft']['mean_diff'], y['S1_minus_ft']['t_p'], y['S1_minus_ft']['wilcoxon_p']))
