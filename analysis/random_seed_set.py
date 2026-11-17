"""Does the 70-label seed set need to be class-stratified? (Section IV-N, CPU, cached features, ~1 min)

    python analysis/random_seed_set.py <artifacts_dir>    -> random_seed_set.json

Compares the published stratified draws (10 leaves per class, rng 42*1000+rep) with 70 leaves drawn from the
training split without regard to class (100 draws, rng 777000+rep). Same probe: multinomial logistic, C = 1."""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ART = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
F = np.load(os.path.join(ART, 'features_dinov1_crop.npz')); Xtr, ytr, Xte, yte, dte = F['Xtr'], F['ytr'], F['Xte'], F['yte'], F['dte']
H = 3
def fit(idx, cw=None):
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight=cw).fit(Xtr[idx], ytr[idx]); cl = list(m.classes_); p = m.predict_proba(Xte)
    return roc_auc_score(dte, 1 - p[:, cl.index(H)]), float((np.array(cl)[p.argmax(1)] == yte).mean())
def strat_draw(r):
    g = np.random.default_rng(42 * 1000 + r)
    return np.concatenate([g.choice(np.where(ytr == c)[0], 10, replace=False) for c in range(7)])
strat = np.array([fit(strat_draw(r)) for r in range(10)])
rnd_idx = [np.random.default_rng(777000 + r).choice(len(ytr), 70, replace=False) for r in range(100)]
rnd = np.array([fit(i) for i in rnd_idx]); rnd_bal = np.array([fit(i, 'balanced') for i in rnd_idx[:20]])
out = dict(stratified_10draws=dict(auc_mean=float(strat[:, 0].mean()), auc_sd=float(strat[:, 0].std(ddof=1)), acc7=float(strat[:, 1].mean())),
           random_100draws=dict(auc_mean=float(rnd[:, 0].mean()), auc_sd=float(rnd[:, 0].std(ddof=1)), auc_p5=float(np.percentile(rnd[:, 0], 5)), acc7=float(rnd[:, 1].mean())),
           random_20draws_balanced_weights=dict(acc7=float(rnd_bal[:, 1].mean())))
json.dump(out, open('random_seed_set.json', 'w'), indent=1); print(json.dumps(out, indent=1))
