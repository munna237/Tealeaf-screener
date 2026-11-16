"""Choose the probe's C by cross-validation INSIDE the 70 labels (no test labels), five seed draws."""
import numpy as np, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
warnings.filterwarnings('ignore')
import os, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'artifacts'
z = np.load(os.path.join(D, 'features_dinov1_crop.npz')); Xtr, ytr, Xte, yte, dte = z['Xtr'], z['ytr'], z['Xte'], z['yte'], z['dte']
GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
def draw(rep, n=10):
    r = np.random.default_rng(42 * 1000 + rep)
    return np.concatenate([r.choice(np.where(ytr == c)[0], n, replace=False) for c in range(7)])
out = {'grid': GRID, 'per_rep': [], 'fixed_C': {}}
for C in (1, 10, 100):
    acc, auc = [], []
    for rep in range(5):
        i = draw(rep); m = LogisticRegression(C=C, max_iter=5000).fit(Xtr[i], ytr[i])
        acc.append(accuracy_score(yte, m.predict(Xte))); auc.append(roc_auc_score(dte, 1 - m.predict_proba(Xte)[:, 3]))
    out['fixed_C'][str(C)] = dict(acc7=[float(np.mean(acc)), float(np.std(acc, ddof=1))], auc=[float(np.mean(auc)), float(np.std(auc, ddof=1))])
for rep in range(5):
    i = draw(rep); X, y = Xtr[i], ytr[i]
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)          # 2 labelled leaves per class per fold
    cvacc = [cross_val_score(LogisticRegression(C=C, max_iter=5000), X, y, cv=cv, scoring='accuracy').mean() for C in GRID]
    best = GRID[int(np.argmax(cvacc))]                                  # ties -> smallest C (strongest regularisation)
    m = LogisticRegression(C=best, max_iter=5000).fit(X, y)
    out['per_rep'].append(dict(rep=rep, cv_acc=[float(a) for a in cvacc], C=best,
                               acc7=float(accuracy_score(yte, m.predict(Xte))), auc=float(roc_auc_score(dte, 1 - m.predict_proba(Xte)[:, 3]))))
a = [r['acc7'] for r in out['per_rep']]; u = [r['auc'] for r in out['per_rep']]
out['cv_selected'] = dict(acc7=[float(np.mean(a)), float(np.std(a, ddof=1))], auc=[float(np.mean(u)), float(np.std(u, ddof=1))], C=[r['C'] for r in out['per_rep']])
json.dump(out, open('probe_C_cv.json', 'w'), indent=1)          
for C, v in out['fixed_C'].items(): print('C=%-4s 7-class %.3f ± %.3f | AUC %.4f ± %.4f' % (C, *v['acc7'], *v['auc']))
print('CV-selected C per draw:', out['cv_selected']['C'])
print('CV-selected: 7-class %.3f ± %.3f | AUC %.4f ± %.4f' % (*out['cv_selected']['acc7'], *out['cv_selected']['auc']))
for r in out['per_rep']: print('  rep', r['rep'], 'C', r['C'], 'cv', [round(x, 3) for x in r['cv_acc']])
