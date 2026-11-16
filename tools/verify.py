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
for script, args in (('reconcile.py', [ART]), ('recalibrate.py', [ART, '0.95']), ('field.py', [ART]), ('probe_C_cv.py', [ART])):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', script), *args],
                       cwd=RES, capture_output=True, text=True)
    check(r.returncode == 0, f'{script} ran' + ('' if r.returncode == 0 else '\n' + r.stderr[-2000:]))
os.replace(os.path.join(RES, 'recalibration_results.json'), os.path.join(RES, 'recal_dup0.95.json'))

print('[3] headline numbers')
rc = json.load(open(os.path.join(RES, 'reconciled_numbers.json')))
ex = json.load(open(os.path.join(ART, 'external_results_v3.json')))
rl = json.load(open(os.path.join(RES, 'recal_dup0.95.json')))
near = lambda a, b, t=5e-4: abs(a - b) <= t
check(near(rc['student']['auc']['mean'], 0.9854), 'deployed student AUC 0.985 (5 seeds)')
check(near(rc['student']['auc']['sd'], 0.0060), 'student AUC SD 0.006')
check(near(rc['student']['acc']['mean'], 0.9661), 'student accuracy 96.6%')
check(near(rc['probe']['auc']['mean'], 0.9853), 'probe (teacher) AUC 0.985')
check(near(rc['probe']['acc_7class']['mean'], 0.6847), '7-class probe accuracy 0.685')
check(near(rc['stats']['paired_t'][1], 0.1265, 1e-3), 'paired t-test p = 0.13')
check(near(rc['stats']['pitman_morgan_p'], 0.0038, 1e-3), 'Pitman-Morgan p = 0.004')
check(near(rc['triage']['field_10%']['cleared']['mean'], 0.847, 1e-3), 'triage: 84.7% cleared at 10% prevalence')
check(rl['EWU']['n'] == 3007 and ex['EWU_box_probe']['n_healthy'] == 905, 'External dataset 1: 3,007 crops (905 healthy)')
check(near(rl['EWU']['zero_shot_fixed']['auc'][0], 0.874, 1e-3), 'External dataset 1: probe zero-shot AUC 0.874')
check(near(ex['EWU_box_onnx_student']['auc'][0], 0.883, 1e-3) and near(ex['EWU_box_onnx_student']['spec'][0], 0.617, 1e-3), 'External dataset 1: deployed AUC 0.883, specificity 0.617')
check(near(rl['EWU']['local_recalibration']['10']['target_only']['auc'][0], 0.973, 1e-3), 'External dataset 1: 10 labels/class recalibrated AUC 0.973 (40 draws)')
check(near(rl['EWU']['local_recalibration']['20']['target_only']['auc'][0], 0.983, 1e-3), 'External dataset 1: 20 labels/class recalibrated AUC 0.983 (40 draws)')
check(near(ex['Ally_whole_onnx_student']['auc'][0], 0.889, 1e-3) and near(ex['Ally_whole_probe']['auc'][0], 0.915, 1e-3), 'External dataset 2: deployed 0.889, probe 0.915')
rg = json.load(open(os.path.join(ART, 'regen_tables_8_9.json')))
check(all(near(rg['table8_clutter'][a]['auc'], v, 1e-3) for a, v in (('0.0', 0.986), ('0.2', 0.979), ('0.4', 0.962), ('0.6', 0.941))), 'Table 8 clutter AUC 0.986 / 0.979 / 0.962 / 0.941')
check(near(rg['table9_shape']['10']['mean'], 0.725, 1e-3) and near(rg['table9_shape']['all']['mean'], 0.792, 1e-3), 'Table 9 shape control 0.725 at 70 labels, 0.792 with all labels')
bt = json.load(open(os.path.join(ART, 'external_blind_transfer.json')))['summary']
check(near(bt['External_1']['S1']['auc'][0], 0.881, 1e-3) and near(bt['External_2']['S1']['auc'][0], 0.931, 1e-3), 'Table 17 blind transfer: S1 AUC 0.881 / 0.931')
check(near(bt['External_1']['S0']['auc'][0], 0.795, 1e-3) and near(bt['External_2']['S0']['auc'][0], 0.851, 1e-3), 'Table 17 blind transfer: S0 AUC 0.795 / 0.851')
fn = json.load(open(os.path.join(RES, 'field_numbers.json')))
fs = fn['summary']
check(near(fn['probe_zero_shot_auc']['whole'][0], 0.962, 1e-3) and near(fn['probe_zero_shot_auc']['sam'][0], 0.970, 1e-3), 'field: zero-shot probe AUC 0.962 (whole) / 0.970 (SAM)')
check(near(fs['S0']['field_whole'][0], 0.719, 1e-3), 'field: S0 (our data only) whole-frame AUC 0.719')
check(near(fs['S1']['field_whole'][0], 0.972, 1e-3) and near(fs['S1']['field_centre'][0], 0.973, 1e-3), 'field: S1 (0 field labels) AUC 0.972 / 0.973')
check(near(fs['S2']['field_whole'][0], 0.989, 1e-3), 'field: S2 (40 field labels) AUC 0.989')
check(near(fs['S1']['tealeaf'][0], 0.983, 1e-3) and near(fs['S0']['tealeaf'][0], 0.986, 1e-3), 'field: in-distribution 0.983 (S1) vs 0.986 (S0)')
check(min(fn['paired_S1_minus_S0_whole']) > 0.09, 'field: S1 beats S0 on every seed (whole frame)')
check(max(fn['max_cosine_to_tealeaf'].values()) < 0.95, 'field: no near-duplicate of TeaLeafDiseaseBD (max cosine < 0.95)')
cv = json.load(open(os.path.join(RES, 'probe_C_cv.json')))
check(near(cv['fixed_C']['1']['acc7'][0], 0.685, 1e-3) and near(cv['fixed_C']['10']['acc7'][0], 0.743, 1e-3) and near(cv['fixed_C']['100']['acc7'][0], 0.752, 1e-3), '7-class accuracy 0.685 / 0.743 / 0.752 at C = 1 / 10 / 100')
check(near(cv['cv_selected']['acc7'][0], 0.743, 1e-3) and near(cv['cv_selected']['auc'][0], 0.990, 1e-3), 'C chosen by CV inside the 70 labels: 7-class 0.743, AUC 0.990')
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
