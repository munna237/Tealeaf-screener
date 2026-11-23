#!/usr/bin/env python3
"""Verify the released artifacts and regenerate every number that does not need images (CPU, ~1 min).

    python tools/verify.py

1. checks SHA-256 of every file listed in artifacts/CHECKSUMS.sha256 (and models/)
2. reruns analysis/reconcile.py and analysis/recalibrate.py into results/
3. asserts the headline numbers reported in the manuscript and response letter"""
import hashlib, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART, RES = os.path.join(ROOT, 'artifacts'), os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)
ok = True
def check(cond, msg):
    global ok
    print(('  PASS  ' if cond else '  FAIL  ') + msg); ok &= bool(cond)

print('[1] checksums')
for line in open(os.path.join(ART, 'CHECKSUMS.sha256')):
    h, rel = line.split()
    p = os.path.join(ROOT, rel)
    check(os.path.exists(p) and hashlib.sha256(open(p, 'rb').read()).hexdigest() == h, rel)

print('[2] regenerate (no images needed)')
for script, args in (('reconcile.py', [ART]), ('recalibrate.py', [ART, '0.95']), ('field.py', [ART]), ('probe_C_cv.py', [ART]), ('ten_draws.py', [ART]), ('random_seed_set.py', [ART]), ('tuned_baselines.py', [ART])):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', script), *args],
                       cwd=RES, capture_output=True, text=True)
    check(r.returncode == 0, f'{script} ran' + ('' if r.returncode == 0 else '\n' + r.stderr[-2000:]))
os.replace(os.path.join(RES, 'recalibration_results.json'), os.path.join(RES, 'recal_dup0.95.json'))

print('[3] numbers as printed in the revised manuscript (lines tagged [superseded] are earlier runs kept for provenance)')
rc = json.load(open(os.path.join(RES, 'reconciled_numbers.json')))
ex = json.load(open(os.path.join(ART, 'external_results_v3.json')))
rl = json.load(open(os.path.join(RES, 'recal_dup0.95.json')))
near = lambda a, b, t=5e-4: abs(a - b) <= t
check(near(rc['student']['auc']['mean'], 0.9854), 'Sec. IV-J: distilled student AUC 0.985 (first 5 draws; abstract uses ten draws, checked below)')
check(near(rc['student']['auc']['sd'], 0.0060), 'student AUC SD 0.006')
check(near(rc['student']['acc']['mean'], 0.9661), 'student accuracy 96.6%')
check(near(rc['probe']['auc']['mean'], 0.9853), 'probe (teacher) AUC 0.985')
check(near(rc['probe']['acc_7class']['mean'], 0.6847), '7-class probe accuracy 0.685')
check(near(rc['stats']['paired_t'][1], 0.1265, 1e-3), 'paired t-test p = 0.13 (first 5 draws)')
check(near(rc['stats']['pitman_morgan_p'], 0.0038, 1e-3), 'Pitman-Morgan p = 0.004 (first 5 draws)')
check(near(rc['triage']['field_10%']['cleared']['mean'], 0.847, 1e-3), 'triage: 84.7% cleared at 10% prevalence')
check(rl['EWU']['n'] == 3007 and ex['EWU_box_probe']['n_healthy'] == 905, 'External dataset 1: 3,007 crops (905 healthy)')
check(near(rl['EWU']['zero_shot_fixed']['auc'][0], 0.874, 1e-3), 'External dataset 1: probe zero-shot AUC 0.874')
check(near(ex['EWU_box_onnx_student']['auc'][0], 0.883, 1e-3) and near(ex['EWU_box_onnx_student']['spec'][0], 0.617, 1e-3), 'External dataset 1: deployed AUC 0.883, specificity 0.617')
check(near(rl['EWU']['local_recalibration']['10']['target_only']['auc'][0], 0.973, 1e-3), 'External dataset 1: 10 labels/class recalibrated AUC 0.973 (40 draws)')
check(near(rl['EWU']['local_recalibration']['20']['target_only']['auc'][0], 0.983, 1e-3), 'External dataset 1: 20 labels/class recalibrated AUC 0.983 (40 draws)')
check(near(ex['Ally_whole_onnx_student']['auc'][0], 0.889, 1e-3) and near(ex['Ally_whole_probe']['auc'][0], 0.915, 1e-3), 'External dataset 2: deployed 0.889, probe 0.915')
rg = json.load(open(os.path.join(ART, 'regen_tables_8_9.json')))
r2 = json.load(open(os.path.join(ART, 'regen_tables_2_8_14.json')))
check(r2['recipe_cosine'][r2['recipe']]['min'] > 0.9999, 'Tables 2, 9, 15 crops reproduce the cached features (cosine 1.000)')
check(all(near(v, 0.9853, 2e-4) for v in (r2['table2']['DINOv1 ViT-S/16']['auc'][0], r2['table8']['0.0']['auc'][0], r2['table14']['clean']['auc'][0])), 'clean 70-label probe AUC 0.985 in Tables 2, 9 and 15 (= Tables 10, 12)')
check(near(r2['table2']['CLIP ViT-B/16']['auc'][0], 0.986, 1e-3) and near(r2['table2']['DINOv2 ViT-S/14']['auc'][0], 0.979, 1e-3), 'Table 2: CLIP 0.986, DINOv2 0.979')
check(all(near(r2['table8'][a]['auc'][0], v, 1e-3) for a, v in (('0.2', 0.979), ('0.4', 0.962), ('0.6', 0.944))), 'Table 9 clutter AUC 0.985 / 0.979 / 0.962 / 0.944')
check(near(r2['table14']['shift']['auc'][0], 0.976, 1e-3) and near(r2['table14']['jitter']['auc'][0], 0.980, 1e-3), 'Table 15: shift 0.976, jitter 0.980')
check(r2['files']['n_all'] == 13084 and abs(r2['timing']['feature_seconds'] - 41) < 1, 'dataset folder holds 13,084 JPEGs; features timed at 41 s (L4)')
check(near(rg['table9_shape']['10']['mean'], 0.725, 1e-3) and near(rg['table9_shape']['all']['mean'], 0.792, 1e-3), 'Table 10 shape control 0.725 at 70 labels, 0.792 with all labels')
bt = json.load(open(os.path.join(ART, 'external_blind_transfer.json')))['summary']
check(near(bt['External_1']['S1']['auc'][0], 0.881, 1e-3) and near(bt['External_2']['S1']['auc'][0], 0.931, 1e-3), '[superseded 3-seed run, not in the paper] blind transfer S1 0.881 / 0.931')
check(near(bt['External_1']['S0']['auc'][0], 0.795, 1e-3) and near(bt['External_2']['S0']['auc'][0], 0.851, 1e-3), '[superseded 3-seed run, not in the paper] blind transfer S0 0.795 / 0.851')
fn = json.load(open(os.path.join(RES, 'field_numbers.json')))
fs = fn['summary']
check(near(fn['probe_zero_shot_auc']['whole'][0], 0.962, 1e-3) and near(fn['probe_zero_shot_auc']['sam'][0], 0.970, 1e-3), 'field: zero-shot probe AUC 0.962 (whole) / 0.970 (SAM)')
check(near(fs['S0']['field_whole'][0], 0.719, 1e-3), '[superseded 3-seed run, not in the paper] field S0 0.719')
check(near(fs['S1']['field_whole'][0], 0.972, 1e-3) and near(fs['S1']['field_centre'][0], 0.973, 1e-3), '[superseded 3-seed run, not in the paper] field S1 0.972 / 0.973')
check(near(fs['S2']['field_whole'][0], 0.989, 1e-3), '[superseded 3-seed run, not in the paper] field S2 0.989')
check(near(fs['S1']['tealeaf'][0], 0.983, 1e-3) and near(fs['S0']['tealeaf'][0], 0.986, 1e-3), '[superseded 3-seed run, not in the paper] field in-distribution 0.983 / 0.986')
check(min(fn['paired_S1_minus_S0_whole']) > 0.09, 'field: S1 beats S0 on every seed (whole frame)')
sc_ = rc['session_check']
check(near(sc_['file_prefix=IMG']['student_auc']['mean'], 0.972, 1e-3) and near(sc_['file_prefix=PXL']['student_auc']['mean'], 0.995, 1e-3), 'Sec. IV-P: within file prefix IMG / PXL, student AUC 0.972 / 0.995')
check(max(fn['max_cosine_to_tealeaf'].values()) < 0.95, 'field: no near-duplicate of TeaLeafDiseaseBD (max cosine < 0.95)')
cv = json.load(open(os.path.join(RES, 'probe_C_cv.json')))
check(near(cv['fixed_C']['1']['acc7'][0], 0.685, 1e-3) and near(cv['fixed_C']['10']['acc7'][0], 0.743, 1e-3) and near(cv['fixed_C']['100']['acc7'][0], 0.752, 1e-3), '7-class accuracy 0.685 / 0.743 / 0.752 at C = 1 / 10 / 100')
check(near(cv['cv_selected']['acc7'][0], 0.743, 1e-3) and near(cv['cv_selected']['auc'][0], 0.990, 1e-3), 'C chosen by CV inside the 70 labels: 7-class 0.743, AUC 0.990')
td = json.load(open(os.path.join(RES, 'ten_draws.json')))
E_, F_, B_, S_ = td['E'], td['F'], td['blind'], td['sam']
check(near(E_['probe_distil']['mean'], 0.978, 1e-3) and near(E_['probe_distil']['sd'], 0.014, 1e-3) and near(E_['distil_acc'], 0.958, 1e-3), 'Table 12, ten draws: student AUC 0.978 +- 0.014, accuracy 95.8%')
check(near(E_['labels_only']['mean'], 0.956, 1e-3) and near(E_['probe']['mean'], 0.980, 1e-3) and near(E_['probe']['sd'], 0.010, 1e-3), 'Table 12, ten draws: labels-only 0.956, probe 0.980 +- 0.010')
check(near(E_['paired_t_p'], 0.022, 1e-3) and near(E_['wilcoxon_p'], 0.0039, 1e-3) and near(E_['pitman_morgan_p'], 0.075, 1e-3), 'Section IV-M: paired t p = 0.022, Wilcoxon p = 0.004, Pitman-Morgan p = 0.075')
check(near(E_['sd_ratio'], 1.88, 1e-2) and near(E_['spearman_gain_vs_labels_only'], -0.733, 1e-3) and near(E_['spearman_student_vs_probe'], 0.939, 1e-3), 'Section IV-M: SD ratio 1.9, rho -0.73 (gain), 0.94 (student vs probe)')
check(near(F_['S0']['whole']['mean'], 0.727, 1e-3) and near(F_['S1']['whole']['mean'], 0.974, 1e-3) and near(F_['S2']['whole']['mean'], 0.988, 1e-3), 'Table 17, ten seeds: field whole-frame AUC S0 0.727 / S1 0.974 / S2 0.988')
check(F_['S1_minus_S0']['wins'] == 10 and near(F_['S1_minus_S0']['wilcoxon_p'], 0.00195, 1e-4), 'Table 17, ten seeds: S1 beats S0 on every seed, Wilcoxon p = 0.002')
check(near(B_['External_1']['S1']['mean'], 0.899, 1e-3) and near(B_['External_2']['S1']['mean'], 0.920, 1e-3) and B_['External_1']['wins'] == 9 and B_['External_2']['wins'] == 7, 'Table 18, ten seeds: S1 0.899 / 0.920; S1 > S0 on 9 and 7 of 10 seeds')
t6 = S_['TeaLeafDiseaseBD']
check(near(t6['area_ratio_mean'], 1.28, 5e-3) and near(t6['area_ratio_median'], 1.68, 5e-3) and near(t6['qc_border_area_ratio_mean'], 1.57, 5e-3), 'Table 6: SAM area ratio on our data 1.28 (mean), 1.68 (median), 1.57 (class-blind QC)')
check(near(t6['shape_only_auc'], 0.904, 1e-3) and near(t6['collapse'], 0.338, 1e-3), 'Table 6: SAM shape-only 0.904, collapse +0.338 on our data')
check(all(near(S_[k]['shape_only_auc'], v, 2e-3) for k, v in (('PlantVillage', 0.768), ('PlantDoc (field)', 0.753), ('RoCoLe', 0.725), ('RiceLeafBD', 0.615))), 'Table 6: SAM shape-only 0.768 / 0.753 / 0.725 / 0.615 on the external datasets')
sf = json.load(open(os.path.join(ART, 'sam_confound_followup.json')))['results']
check([sf[k]['primary']['algorithm2'] for k in ('TeaLeafDiseaseBD', 'PlantVillage', 'PlantDoc (field)', 'RoCoLe', 'RiceLeafBD')] == ['Confound present', 'Inconclusive; report both', 'Inconclusive; report both', 'Signal is pathology', 'Signal is pathology'], 'Table 6: Algorithm 2 verdicts')
tb = json.load(open(os.path.join(RES, 'tuned_baselines.json'))); T12, TR = tb['table12'], tb['transfer']
bt_ = json.load(open(os.path.join(ART, 'baseline_tuned.json')))
check(near(T12['probe_distil']['mean'], td['E']['probe_distil']['mean'], 1e-6) and near(T12['labels_only_student_recipe']['mean'], td['E']['labels_only']['mean'], 1e-6), 'baseline_tuned.json uses the same ten draws as Experiment E (0.978 / 0.956 reproduce)')
check(near(T12['ft_lr1e-4_strongaug']['mean'], 0.975, 1e-3) and near(T12['ft_lr3e-4_strongaug']['mean'], 0.976, 1e-3) and T12['ft_lr1e-4_strongaug']['distil_minus_ft']['t_p'] > 0.5 and T12['ft_lr3e-4_strongaug']['distil_minus_ft']['t_p'] > 0.5, 'Table 12: tuned fine-tunes 0.975 / 0.976, not different from probe + distillation (t p = 0.53 / 0.65)')
check(near(T12['ft_lr1e-4_strongaug']['pitman_morgan_p'], 0.16, 1e-2) and near(T12['ft_lr3e-4_strongaug']['pitman_morgan_p'], 0.65, 1e-2), 'Sec. IV-M: no variance advantage over tuned fine-tunes (Pitman-Morgan p = 0.16 / 0.65)')
check(near(T12['probe_imagenet_mobilenet_v3_small']['mean'], 0.960, 1e-3) and near(T12['probe_imagenet_resnet50']['mean'], 0.961, 1e-3) and T12['all_four_added_survive'] and all(T12[k]['dino_minus_imagenet']['wins'] == 10 for k in ('probe_imagenet_mobilenet_v3_small', 'probe_imagenet_resnet50')), 'Table 12 / Table 2 note: DINO probe beats ImageNet probes (0.960 / 0.961) on 10/10 draws, survives Bonferroni (0.0125)')
check(near(TR['ft_lr1e-4_strongaug']['whole']['mean'], 0.857, 1e-3) and near(TR['ft_lr3e-4_strongaug']['whole']['mean'], 0.836, 1e-3) and all(TR[k][v]['S1_minus_ft']['wins'] == 10 for k in TR for v in ('whole', 'centre')), 'Table 17: tuned fine-tunes on the field 0.857 / 0.836 (whole); S1 better on 10/10 draws, whole and centre')
check(near(TR['ft_lr1e-4_strongaug']['External_1']['mean'], 0.890, 1e-3) and near(TR['ft_lr3e-4_strongaug']['External_2']['mean'], 0.905, 1e-3) and all(TR[k][e]['S1_minus_ft']['t_p'] > 0.05 for k in TR for e in ('External_1', 'External_2')), 'Table 18: tuned fine-tunes 0.890 / 0.912 and 0.861 / 0.905 externally; no significant difference from S1')
check(max(TR[k]['tealeaf_max_abs_diff_vs_table12'] for k in TR) < 0.006, 'Table 17 note: retrained fine-tunes within 0.006 of Table 12 on our test split')
rs = json.load(open(os.path.join(RES, 'random_seed_set.json')))
check(near(rs['random_100draws']['auc_mean'], 0.981, 1e-3) and near(rs['random_100draws']['auc_p5'], 0.964, 1e-3) and near(rs['stratified_10draws']['auc_mean'], 0.980, 1e-3), 'Sec. IV-N: 70 random (unstratified) labels AUC 0.981, 5th pct 0.964, vs stratified 0.980')
c7 = json.load(open(os.path.join(ART, 'controls_70labels.json')))
check(near(c7['feat_resid']['mean'], 0.971, 1e-3) and near(c7['feat_pairs']['mean'], 0.970, 1e-3) and near(c7['shape_pairs']['mean'], 0.510, 1e-3) and c7['n_pairs'] == 510, 'Table 8, 70 labels: residualised 0.971, matched pairs 0.970 (shape 0.510, 510 pairs)')
fr = json.load(open(os.path.join(ART, 'export_report_field_S1.json')))
fp = os.path.join(ROOT, 'models', 'tea_screener_field_S1.onnx')
if os.path.exists(fp):
    check(hashlib.sha256(open(fp, 'rb').read()).hexdigest() == fr['sha256'], 'models/tea_screener_field_S1.onnx is the file in export_report_field_S1.json')
else:
    check(False, 'models/tea_screener_field_S1.onnx missing (copy tea_screener_field_S1_rep1.onnx from Drive)')
xr = json.load(open(os.path.join(ART, 'export_report.json')))
h = hashlib.sha256(open(os.path.join(ROOT, 'models', 'tea_screener.onnx'), 'rb').read()).hexdigest()
check(h == xr['sha256'], 'models/tea_screener.onnx is the file described in export_report.json')
check(near(xr['test_auc'], 0.9860, 1e-3) and xr['opset'] == 13, 'exported model: AUC 0.986, opset 13')
try:
    import numpy as np, onnxruntime as ort
    z = ort.InferenceSession(os.path.join(ROOT, 'models', 'tea_screener.onnx')).run(None, {'input': np.zeros((2, 3, 224, 224), np.float32)})[0]
    check(z.shape == (2, 2), 'ONNX model loads and runs with dynamic batch')
except ImportError:
    print('  SKIP  onnxruntime not installed')
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED'); sys.exit(0 if ok else 1)
