#!/usr/bin/env python3
"""Run in Colab with Drive mounted: copies the ORIGINAL experiment scripts and notebooks into this repo,
strips notebook outputs (they can be large and may contain local paths), and reports anything missing.

    !python tools/collect_from_drive.py --drive /content/drive/MyDrive/TeaLeaf_DINOv2_Study --repo .

Nothing here is rewritten: files are copied verbatim (notebooks: outputs cleared only)."""
import argparse, glob, json, os, shutil
SCRIPTS = ['leakage_audit.py', 'shape_confound.py', 'feature_shape_control.py', 'few_shot_*.py',
           'clutter_robustness_fast.py', 'fig_diagnostics.py', 'triage_analysis.py']
NOTEBOOKS = ['Tea_Deployment*.ipynb', 'Tea_leaf_figure*.ipynb', 'TEA_LEAF_FIGURES*.ipynb', 'Tea_Leaf.ipynb',
             'Tea_Leaf_Re_*.ipynb', 'Audit*.ipynb', 'Multi_seed*.ipynb', '*revision*.ipynb']
def strip(src, dst):
    nb = json.load(open(src))
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'code': c['outputs'], c['execution_count'] = [], None
    json.dump(nb, open(dst, 'w'), indent=1)
ap = argparse.ArgumentParser(); ap.add_argument('--drive', required=True); ap.add_argument('--repo', default='.')
ap.add_argument('--extra', nargs='*', default=[], help='more folders to search (e.g. /content/drive/MyDrive/Colab Notebooks)')
a = ap.parse_args(); roots = [a.drive, *a.extra]
def find(pat):
    return sorted({p for r in roots for p in glob.glob(os.path.join(r, '**', pat), recursive=True)})
missing = []
for pat in SCRIPTS:
    hits = find(pat)
    if not hits: missing.append(pat)
    for h in hits: shutil.copy(h, os.path.join(a.repo, 'scripts', 'original', os.path.basename(h))); print('script  ', h)
for pat in NOTEBOOKS:
    hits = find(pat)
    if not hits: missing.append(pat)
    for h in hits:
        strip(h, os.path.join(a.repo, 'notebooks', 'original', os.path.basename(h).replace(' ', '_'))); print('notebook', h)
print('\nMISSING (add by hand or pass --extra):', missing or 'none')
print('Before pushing: open each notebook and remove any tokens, API keys, or personal paths.')
