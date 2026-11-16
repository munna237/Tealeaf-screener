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
for script, args in (('reconcile.py', [ART]), ('recalibrate.py', [ART, '0.95']), ('field.py', [ART])):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'analysis', script), *args],
                       cwd=RES, capture_output=True, text=True)
    check(r.returncode == 0, f'{script} ran' + ('' if r.returncode == 0 else '\n' + r.stderr[-2000:]))
os.replace(os.path.join(RES, 'recalibration_results.json'), os.path.join(RES, 'recal_dup0.95.json'))

print('[3] headline numbers')
rc = json.load(open(os.path.join(RES, 'reconciled_numbers.json')))
ex = json.load(open(os.path.join(ART, 'external_results.json')))
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
check(near(rl['EWU']['zero_shot_fixed']['auc'][0], ex['EWU_box_probe']['auc'][0], 1e-3), 'EWU zero-shot AUC 0.873')
check(near(rl['EWU']['local_recalibration']['10']['target_only']['auc'][0], 0.977, 1e-3), 'EWU 10/class recalibrated AUC 0.977')
fn = json.load(open(os.path.join(RES, 'field_numbers.json')))
fs = fn['summary']
check(near(fn['probe_zero_shot_auc']['whole'][0], 0.962, 1e-3) and near(fn['probe_zero_shot_auc']['sam'][0], 0.970, 1e-3), 'field: zero-shot probe AUC 0.962 (whole) / 0.970 (SAM)')
check(near(fs['S0']['field_whole'][0], 0.719, 1e-3), 'field: S0 (our data only) whole-frame AUC 0.719')
check(near(fs['S1']['field_whole'][0], 0.972, 1e-3) and near(fs['S1']['field_centre'][0], 0.973, 1e-3), 'field: S1 (0 field labels) AUC 0.972 / 0.973')
check(near(fs['S2']['field_whole'][0], 0.989, 1e-3), 'field: S2 (40 field labels) AUC 0.989')
check(near(fs['S1']['tealeaf'][0], 0.983, 1e-3) and near(fs['S0']['tealeaf'][0], 0.986, 1e-3), 'field: in-distribution 0.983 (S1) vs 0.986 (S0)')
check(min(fn['paired_S1_minus_S0_whole']) > 0.09, 'field: S1 beats S0 on every seed (whole frame)')
check(max(fn['max_cosine_to_tealeaf'].values()) < 0.95, 'field: no near-duplicate of TeaLeafDiseaseBD (max cosine < 0.95)')
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
