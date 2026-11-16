"""
clutter_robustness_fast.py -- same experiment, ~15x faster, with progress bars.

WHAT WAS SLOW
The previous version ran ONE image per DINO forward pass:
    3 repeats x 6 alphas x 1200 images = 21,600 single-image GPU calls.
A ViT forward on batch=1 wastes almost the entire GPU.

WHAT CHANGED
  1. DINO forwards are BATCHED (64 at a time). This is the dominant speedup.
  2. Source images are decoded ONCE into memory (resized to 224) and reused
     across every alpha and repeat -- previously each was re-read from disk 18x.
  3. The evaluation subset is fixed across alphas within a repeat, so the SAME
     leaves are compared at every clutter level (this is also more correct: it
     makes the comparison paired rather than between independent samples).
  4. tqdm progress bars on decoding, on each (repeat, alpha) pass, and overall.

The experiment is unchanged; only the plumbing is faster and the comparison is
now properly paired.
"""

import os
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
def field_textures(n=40, size=224, seed=0):
    """Deterministic synthetic soil/foliage textures."""
    import cv2
    rng = np.random.RandomState(seed)
    out = []
    palettes = [
        [(34, 85, 34), (60, 110, 45), (20, 60, 25)],     # foliage
        [(85, 65, 34), (110, 90, 55), (60, 45, 25)],     # soil
        [(45, 90, 45), (70, 120, 60), (30, 70, 35)],     # grass
        [(70, 60, 40), (95, 85, 60), (50, 40, 30)],      # dry litter
    ]
    for i in range(n):
        pal = palettes[i % len(palettes)]
        img = np.zeros((size, size, 3), np.uint8)
        img[:] = pal[0]
        for _ in range(70):
            x, y = rng.randint(0, size - 20, 2)
            w, h = rng.randint(15, 70, 2)
            col = pal[rng.randint(len(pal))]
            col = np.clip(np.array(col) + rng.randint(-30, 30, 3), 0, 255)
            img[y:y + h, x:x + w] = col
        img = cv2.GaussianBlur(img, (9, 9), 3)
        out.append(img.astype(np.float32))          # keep as array: blending is faster
    return np.stack(out)


# ---------------------------------------------------------------------------
def _batched_features(imgs, model, tf, device, batch=64, desc=""):
    """imgs: list of PIL images (or None). Returns [N, D] with NaN rows for None."""
    idx_ok = [i for i, im in enumerate(imgs) if im is not None]
    if not idx_ok:
        return np.full((len(imgs), 384), np.nan, dtype=np.float32)

    feats = np.full((len(imgs), 384), np.nan, dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, len(idx_ok), batch), desc=desc, leave=False):
            chunk = idx_ok[s:s + batch]
            xb = torch.stack([tf(imgs[i]) for i in chunk]).to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                f = model(xb)
            f = f.float().cpu().numpy()
            f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
            feats[chunk] = f
    return feats


def _prep_at_alpha(base_imgs, bgs, bg_choice, alpha, mask_fn, crop_fn):
    """Blend -> segment -> crop. Returns (list of PIL crops or None, seg_ok array)."""
    crops, ok = [], []
    for j, arr in enumerate(base_imgs):
        if alpha > 0:
            b = bgs[bg_choice[j] % len(bgs)]
            a = ((1 - alpha) * arr + alpha * b).clip(0, 255).astype(np.uint8)
        else:
            a = arr.astype(np.uint8)
        img = Image.fromarray(a)
        m = mask_fn(img)
        if m is None:
            crops.append(None); ok.append(False); continue
        c = crop_fn(img, m)
        crops.append(c); ok.append(c is not None)
    return crops, np.array(ok)


# ---------------------------------------------------------------------------
def run(records, model, tf, device, mask_fn, crop_fn, shape_fn,
        y_bin, y_cls, class_names, healthy_idx,
        alphas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.6),
        n_shot=10, n_eval=1200, n_repeats=3, seed=0,
        batch=64, results_dir=None, verbose=True):

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score

    K = len(class_names)
    N = len(records)
    bgs = field_textures()
    results = {a: {"auc": [], "acc": [], "seg": []} for a in alphas}

    for rep in range(n_repeats):
        print(f"\n--- repeat {rep+1}/{n_repeats} ---")
        r = np.random.default_rng(seed * 100 + rep)

        # ---- expert labels n_shot per class (on CLEAN images) ----
        seed_idx, ok = [], True
        for c in range(K):
            pool = np.where(y_cls == c)[0]
            if len(pool) < n_shot:
                ok = False; break
            seed_idx.append(r.choice(pool, n_shot, replace=False))
        if not ok:
            continue
        seed_idx = np.concatenate(seed_idx)

        rest = np.setdiff1d(np.arange(N), seed_idx)
        eval_idx = r.choice(rest, min(n_eval, len(rest)), replace=False)
        bg_choice = r.integers(0, len(bgs), len(eval_idx))

        # ---- decode source images ONCE (reused for every alpha) ----
        def _decode(idxs, desc):
            out = []
            for i in tqdm(idxs, desc=desc, leave=False):
                try:
                    im = Image.open(records[i]["path"]).convert("RGB").resize((224, 224))
                    out.append(np.asarray(im).astype(np.float32))
                except Exception:
                    out.append(np.zeros((224, 224, 3), np.float32))
            return out

        seed_imgs = _decode(seed_idx, "decode seed")
        eval_imgs = _decode(eval_idx, "decode eval")

        # ---- train the probe on CLEAN seed crops ----
        s_crops, s_ok = _prep_at_alpha(seed_imgs, bgs,
                                       np.zeros(len(seed_idx), int), 0.0,
                                       mask_fn, crop_fn)
        Fs = _batched_features(s_crops, model, tf, device, batch, "features (seed)")
        keep = ~np.isnan(Fs[:, 0])
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        probe.fit(Fs[keep], y_bin[seed_idx][keep])

        # ---- evaluate at each clutter level (same leaves every time) ----
        for a in alphas:
            crops, seg = _prep_at_alpha(eval_imgs, bgs, bg_choice, a,
                                        mask_fn, crop_fn)
            F = _batched_features(crops, model, tf, device, batch,
                                  f"features (a={a})")
            good = ~np.isnan(F[:, 0])
            if good.sum() < 50:
                results[a]["auc"].append(np.nan)
                results[a]["acc"].append(np.nan)
                results[a]["seg"].append(float(seg.mean()))
                continue
            ye = y_bin[eval_idx][good]
            p = probe.predict_proba(F[good])[:, 1]
            au = float(roc_auc_score(ye, p))
            ac = float(accuracy_score(ye, (p > 0.5).astype(int)))
            results[a]["auc"].append(au)
            results[a]["acc"].append(ac)
            results[a]["seg"].append(float(seg.mean()))
            print(f"   alpha={a:<4.1f}  AUC {au:.4f}  acc {ac:.3f}  "
                  f"seg {seg.mean()*100:.0f}%")

    # ---- report ----
    print("\n" + "=" * 76)
    print("CLUTTER ROBUSTNESS OF THE 70-LABEL DEPLOYED MODEL")
    print("  probe trained on clean images; evaluated on the SAME leaves under clutter")
    print("=" * 76)
    print(f"{'alpha':>6s} {'condition':<26s} {'AUC':>15s} {'acc':>8s} {'seg ok':>8s}")
    print("-" * 76)
    labels = {0.0: "protocol followed", 0.1: "slight backdrop texture",
              0.2: "imperfect backdrop", 0.3: "poor backdrop",
              0.4: "protocol abandoned", 0.6: "full field clutter"}
    for a in alphas:
        au = np.nanmean(results[a]["auc"]); su = np.nanstd(results[a]["auc"])
        ac = np.nanmean(results[a]["acc"]); sg = np.nanmean(results[a]["seg"])
        print(f"{a:>6.1f} {labels.get(a,''):<26s} "
              f"{au:>7.4f}+/-{su:.3f} {ac:>8.3f} {sg*100:>7.0f}%")
    print("-" * 76)
    base = np.nanmean(results[alphas[0]]["auc"])
    limit = None
    for a in alphas[1:]:
        if np.nanmean(results[a]["auc"]) < base - 0.05:
            limit = a; break
    if limit:
        print(f"  OPERATING LIMIT: performance degrades materially at alpha >= {limit}")
        print(f"  (AUC {np.nanmean(results[limit]['auc']):.3f} vs {base:.3f} clean)")
    else:
        print("  Model is robust across every clutter level tested.")
    print("  NB: alpha degrades the WHOLE image, leaf included -- harsher than")
    print("      background clutter alone. It bounds the failure mode; it does")
    print("      not substitute for a field trial.")
    print("=" * 76)

    out = {str(a): {"auc": float(np.nanmean(results[a]["auc"])),
                    "auc_std": float(np.nanstd(results[a]["auc"])),
                    "acc": float(np.nanmean(results[a]["acc"])),
                    "seg_rate": float(np.nanmean(results[a]["seg"]))}
           for a in alphas}
    out["operating_limit_alpha"] = limit
    if results_dir:
        import json
        with open(os.path.join(results_dir, "clutter_robustness.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print("saved clutter_robustness.json")
    return out
