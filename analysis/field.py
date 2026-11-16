#!/usr/bin/env python3
"""Field-condition evaluation (manuscript Section IV-Q, Table 16) from released artifacts, no images.

    python analysis/field.py artifacts/        -> field_numbers.json

Recomputes from per-image scores and frozen features:
  * AUC of every Experiment F student (S0/S1/S2, three seeds, prevalence 16/26/36%) on the held-out field half
  * zero-shot 70-label probe AUC on the 2,500 field photographs (whole frame, SAM crop)
  * where field Red Rust and Blight land in the seven-class probe output
  * maximum DINO cosine between field photographs and TeaLeafDiseaseBD (near-duplicate check)
Field data: Chowdhury & Arman (2025), Mendeley Data, doi:10.17632/3x42rbj8yv.1 (CC BY 4.0)."""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ART = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
HEALTHY, RED_SPIDER = 3, 6
out = {}

# Experiment F students
S = np.load(os.path.join(ART, 'experiment_F_scores.npz'), allow_pickle=True)
fd, td = S['field_test_d'], S['tealeaf_test_d']
runs = sorted(k[:-len('_tealeaf')] for k in S.files if k.endswith('_tealeaf') and k.startswith(('S0_', 'S1_', 'S2_')))
out['students'] = {r: {'field_whole': roc_auc_score(fd, S[r + '_field_whole']),
                       'field_centre': roc_auc_score(fd, S[r + '_field_centre']),
                       'tealeaf': roc_auc_score(td, S[r + '_tealeaf'])} for r in runs}
agg = {}
for s in ('S0', 'S1', 'S2'):
    rr = [r for r in runs if r.startswith(s + '_rep')]
    agg[s] = {m: [float(np.mean([out['students'][r][m] for r in rr])),
                  float(np.std([out['students'][r][m] for r in rr], ddof=1))] for m in ('field_whole', 'field_centre', 'tealeaf')}
out['summary'] = agg
out['paired_S1_minus_S0_whole'] = [out['students'][f'S1_rep{i}']['field_whole'] - out['students'][f'S0_rep{i}']['field_whole'] for i in range(3)]

# zero-shot probe, five 70-label draws (seed rng(42*1000+rep), C=1)
Z = np.load(os.path.join(ART, 'features_dinov1_crop.npz'))
Xtr, ytr, Xte, yte = Z['Xtr'], Z['ytr'], Z['Xte'], Z['yte']
F = {t: np.load(os.path.join(ART, f'feat_Field_{t}.npz'), allow_pickle=True) for t in ('whole', 'sam')}
zs = {t: [] for t in F}; zs['in_distribution'] = []
share = {(t, c): [] for t in F for c in ('Red Rust', 'Blight')}
for rep in range(5):
    rng = np.random.default_rng(42 * 1000 + rep)
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], 10, replace=False) for c in np.unique(ytr)])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr[idx], ytr[idx])
    zs['in_distribution'].append(roc_auc_score(yte != HEALTHY, 1 - m.predict_proba(Xte)[:, HEALTHY]))
    for t, D in F.items():
        zs[t].append(roc_auc_score(D['cls'] != 'Healthy', 1 - m.predict_proba(D['X'])[:, HEALTHY]))
        pred = m.predict(D['X'])
        for c in ('Red Rust', 'Blight'):
            share[(t, c)].append(float((pred[D['cls'] == c] == RED_SPIDER).mean()))
out['probe_zero_shot_auc'] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in zs.items()}
out['to_red_spider'] = {f'{t}/{c}': {'mean': float(np.mean(v)), 'min': min(v), 'max': max(v)} for (t, c), v in share.items()}

# near-duplicate check against TeaLeafDiseaseBD
A = np.vstack([Xtr, Xte]).astype(float); A /= np.linalg.norm(A, axis=1, keepdims=True)
out['max_cosine_to_tealeaf'] = {}
for t, D in F.items():
    B = D['X'].astype(float); B /= np.linalg.norm(B, axis=1, keepdims=True)
    out['max_cosine_to_tealeaf'][t] = float((B @ A.T).max())

json.dump(out, open('field_numbers.json', 'w'), indent=1)
for s, v in agg.items():
    print(f"{s}: whole {v['field_whole'][0]:.3f}±{v['field_whole'][1]:.3f}  centre {v['field_centre'][0]:.3f}±{v['field_centre'][1]:.3f}  ours {v['tealeaf'][0]:.3f}±{v['tealeaf'][1]:.3f}")
print('probe zero-shot', {k: round(v[0], 3) for k, v in out['probe_zero_shot_auc'].items()})
print('to Red Spider', {k: round(v['mean'], 2) for k, v in out['to_red_spider'].items()})
print('max cosine', out['max_cosine_to_tealeaf'])
