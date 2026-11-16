"""
leakage_audit.py  -- Train/test leakage audit for the Tea Leaf pipeline.

Detects exact + near-duplicate images (including flip / 90-rotation / colour-jitter
/ re-encode augmentation siblings) and reports how many TEST images have a
duplicate sitting in the TRAIN set under the exact split you used
(random.seed(42), 20% test). Then offers a group-aware split that keeps duplicate
clusters on one side.

Pure numpy + PIL. No pip installs needed. Optional DINOv1 embedding pass reuses
your already-loaded `dino_v1` if present in globals.

Interpreting the headline number
  leaked_test_fraction = (test images with >=1 duplicate in train) / (test images)
    < 1%    : split is essentially clean, your AUCs are trustworthy.
    1-5%    : minor, note it and keep going.
    > 5-10% : material inflation; re-run key results with the grouped split.
    > 20%   : the ~1.0 supervised AUC is explained. Grouped split is mandatory.
"""

import os, json, random, hashlib
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# 1. Perceptual hashes (self-contained, numpy only)
# --------------------------------------------------------------------------
def _dct_matrix(n):
    x = np.arange(n)
    k = x.reshape(-1, 1)
    d = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    d[0] *= 1.0 / np.sqrt(2)
    return d * np.sqrt(2.0 / n)

_DCT32 = _dct_matrix(32)

def _phash_bits(gray32):
    """32x32 grayscale float -> 64-bit pHash (top-left 8x8 DCT, median threshold)."""
    dct = _DCT32 @ gray32 @ _DCT32.T
    block = dct[:8, :8].copy()
    med = np.median(block[1:].flatten())   # drop DC term from the median
    return (block > med).flatten()          # 64 booleans

def _dhash_bits(gray):
    """dHash: 9x8 grayscale, compare adjacent columns -> 64 bits."""
    g = np.asarray(Image.fromarray(gray).resize((9, 8), Image.BILINEAR), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).flatten()

def _orient_variants(img_gray):
    """8 dihedral orientations of a grayscale PIL image."""
    outs = []
    for base in (img_gray, img_gray.transpose(Image.FLIP_LEFT_RIGHT)):
        for rot in (0, 90, 180, 270):
            outs.append(base.rotate(rot, expand=False))
    return outs

def canonical_hashes(path):
    """
    Return (phash_bits, dhash_bits) canonicalised over the 8 flip/rotation
    orientations, so an image and its flipped/90-rotated augmentation collapse
    to the same hash. Returns None on read failure.
    """
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    variants = _orient_variants(img)
    # pHash canonical: min over orientations (as packed uint64)
    ph_all, dh_all = [], []
    for v in variants:
        g32 = np.asarray(v.resize((32, 32), Image.BILINEAR), dtype=np.float32)
        ph_all.append(_phash_bits(g32))
        dh_all.append(_dhash_bits(np.asarray(v, dtype=np.uint8)))
    ph_all = np.array(ph_all)   # [8,64] bool
    dh_all = np.array(dh_all)
    ph_int = _bits_to_uint64(ph_all)
    dh_int = _bits_to_uint64(dh_all)
    ph_canon = ph_all[np.argmin(ph_int)]
    dh_canon = dh_all[np.argmin(dh_int)]
    return ph_canon, dh_canon

def _bits_to_uint64(bits2d):
    """[.,64] bool -> uint64 per row (for canonical min + fast equality)."""
    powers = (1 << np.arange(64, dtype=np.uint64))[::-1]
    return (bits2d.astype(np.uint64) * powers).sum(axis=1)


# --------------------------------------------------------------------------
# 2. Near-duplicate edges via chunked Hamming distance
# --------------------------------------------------------------------------
def _hamming_edges(bits, thresh, chunk=128):
    """
    bits: [N,64] uint8 (0/1). Return list of (i,j) i<j with hamming <= thresh.
    Chunked so memory stays ~ chunk*N*64 bytes.
    """
    n = bits.shape[0]
    edges = []
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = bits[i0:i1]                                   # [c,64]
        # hamming to ALL rows: [c,N]
        dist = (block[:, None, :] ^ bits[None, :, :]).sum(axis=2)
        rows, cols = np.where(dist <= thresh)
        for r, c in zip(rows, cols):
            gi = i0 + r
            if gi < c:                                        # i<j, skip self & dups
                edges.append((int(gi), int(c)))
    return edges


# --------------------------------------------------------------------------
# 3. Union-find
# --------------------------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# --------------------------------------------------------------------------
# 4. Build the image list EXACTLY like the training code did
# --------------------------------------------------------------------------
def build_image_list(data_root, healthy_substr="healthy"):
    classes = sorted(d for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    imgs = []
    for c in classes:
        cdir = os.path.join(data_root, c)
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append({
                    "path": os.path.join(cdir, f),
                    "class": c,
                    "class_idx": cls_to_idx[c],
                    "is_diseased": 0 if healthy_substr in c.lower() else 1,
                })
    return imgs, classes


def reconstruct_split(imgs, seed=42, test_frac=0.2):
    """Same recipe as the recovery cell: shuffle all, first test_frac -> test."""
    random.seed(seed)
    shuf = imgs.copy()
    random.shuffle(shuf)
    n_test = int(test_frac * len(shuf))
    test_set = {id(x) for x in shuf[:n_test]}
    # return boolean is_test aligned to the ORIGINAL imgs order
    return [id(x) in test_set for x in imgs]


# --------------------------------------------------------------------------
# 5. Main audit
# --------------------------------------------------------------------------
def run_audit(data_root,
              seed=42, test_frac=0.2,
              phash_thresh=6, dhash_thresh=6,
              use_embeddings=False, emb_cos_thresh=0.96,
              out_json=None, save_montage_dir=None,
              max_montage_clusters=12):
    imgs, classes = build_image_list(data_root)
    n = len(imgs)
    print(f"Images: {n} across {len(classes)} classes")
    print("Hashing (canonical over 8 orientations)...")

    md5 = [None] * n
    ph = np.zeros((n, 64), dtype=np.uint8)
    dh = np.zeros((n, 64), dtype=np.uint8)
    ok = np.ones(n, dtype=bool)

    for i, rec in enumerate(imgs):
        try:
            with open(rec["path"], "rb") as fh:
                md5[i] = hashlib.md5(fh.read()).hexdigest()
        except Exception:
            md5[i] = None
        h = canonical_hashes(rec["path"])
        if h is None:
            ok[i] = False
            continue
        ph[i], dh[i] = h[0].astype(np.uint8), h[1].astype(np.uint8)
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i+1}/{n}")

    uf = _UF(n)          # full clustering (all evidence)
    uf_exact = _UF(n)    # md5 + hash evidence only (NO embeddings) -> high-confidence dups

    def _union_both(a, b):
        uf.union(a, b); uf_exact.union(a, b)

    # (a) exact file duplicates (md5)
    md5_groups = {}
    for i, m in enumerate(md5):
        if m:
            md5_groups.setdefault(m, []).append(i)
    exact_pairs = 0
    for g in md5_groups.values():
        for j in g[1:]:
            _union_both(g[0], j); exact_pairs += 1

    # (b) canonical-hash exact matches (flip/rotate/colour/re-encode siblings)
    def _exact_hash_union(bits):
        d = {}
        for i in range(n):
            if not ok[i]:
                continue
            key = bits[i].tobytes()
            d.setdefault(key, []).append(i)
        cnt = 0
        for g in d.values():
            for j in g[1:]:
                _union_both(g[0], j); cnt += 1
        return cnt
    ph_exact = _exact_hash_union(ph)
    dh_exact = _exact_hash_union(dh)

    # (c) near-duplicate edges (hamming <= thresh)
    print("Near-duplicate scan (chunked Hamming)...")
    valid = np.where(ok)[0]
    ph_edges = _hamming_edges(ph[valid], phash_thresh)
    dh_edges = _hamming_edges(dh[valid], dhash_thresh)
    for a, b in ph_edges:
        _union_both(valid[a], valid[b])
    for a, b in dh_edges:
        _union_both(valid[a], valid[b])

    # (d) optional embedding pass (reuse DINOv1 from caller globals) -> full uf only
    emb_edges = 0
    emb_edge_list = []
    if use_embeddings:
        emb_edges, emb_edge_list = _embedding_pass(imgs, uf, emb_cos_thresh)

    # ---- assemble clusters ----
    clusters = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    multi = {r: g for r, g in clusters.items() if len(g) > 1}
    imgs_in_dup = sum(len(g) for g in multi.values())

    # cross-class clusters (label-consistency red flag)
    cross_class = 0
    for g in multi.values():
        if len({imgs[i]["class"] for i in g}) > 1:
            cross_class += 1

    # ---- leakage under the real split ----
    is_test = reconstruct_split(imgs, seed=seed, test_frac=test_frac)
    n_test = sum(is_test)
    leaked_test = 0
    leaked_examples = []
    for g in multi.values():
        has_test = any(is_test[i] for i in g)
        has_train = any(not is_test[i] for i in g)
        if has_test and has_train:
            for i in g:
                if is_test[i]:
                    leaked_test += 1
            if len(leaked_examples) < 20:
                leaked_examples.append([imgs[i]["path"] for i in g])
    leak_frac = leaked_test / max(1, n_test)

    # ---- report ----
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT REPORT")
    print("=" * 60)
    print(f"Total images                 : {n}")
    print(f"Unreadable / skipped         : {int((~ok).sum())}")
    print(f"Exact file dups (md5 pairs)  : {exact_pairs}")
    print(f"Canonical pHash exact unions : {ph_exact}")
    print(f"Canonical dHash exact unions : {dh_exact}")
    print(f"pHash near-dup edges (<= {phash_thresh}) : {len(ph_edges)}")
    print(f"dHash near-dup edges (<= {dhash_thresh}) : {len(dh_edges)}")
    if use_embeddings:
        print(f"Embedding edges (cos>={emb_cos_thresh}) : {emb_edges}")
    print("-" * 60)
    print(f"Duplicate clusters (size>=2) : {len(multi)}")
    print(f"Images inside a dup cluster  : {imgs_in_dup} "
          f"({100*imgs_in_dup/n:.1f}% of dataset)")
    print(f"Cross-CLASS dup clusters     : {cross_class}  "
          f"(these are also label-noise)")
    print("-" * 60)
    print(f"Split: seed={seed}, test_frac={test_frac}  ->  test={n_test}")
    print(f"TEST images with a TRAIN duplicate : {leaked_test}")
    print(f"LEAKED TEST FRACTION               : {leak_frac*100:.2f}%")
    print("=" * 60)
    verdict = ("CLEAN" if leak_frac < 0.01 else
               "MINOR" if leak_frac < 0.05 else
               "MATERIAL" if leak_frac < 0.20 else "SEVERE")
    print(f"VERDICT: {verdict}")
    if verdict in ("MATERIAL", "SEVERE"):
        print(" -> Re-run headline results with grouped_split() below.")
    print("=" * 60)

    # cluster-size histogram
    from collections import Counter
    sizes = Counter(len(g) for g in multi.values())
    if sizes:
        print("Cluster-size distribution:")
        for s in sorted(sizes):
            print(f"  size {s:>3d}: {sizes[s]} clusters")

    if save_montage_dir:
        _save_montages(multi, imgs, save_montage_dir, max_montage_clusters)

    report = {
        "n_images": n, "n_classes": len(classes),
        "exact_md5_pairs": exact_pairs,
        "phash_exact_unions": ph_exact, "dhash_exact_unions": dh_exact,
        "phash_near_edges": len(ph_edges), "dhash_near_edges": len(dh_edges),
        "dup_clusters": len(multi), "images_in_dup_clusters": imgs_in_dup,
        "pct_images_in_dups": round(100 * imgs_in_dup / n, 2),
        "cross_class_clusters": cross_class,
        "split_seed": seed, "test_frac": test_frac, "n_test": n_test,
        "leaked_test_images": leaked_test,
        "leaked_test_fraction": round(leak_frac, 4),
        "verdict": verdict,
        "leaked_examples": leaked_examples[:20],
    }
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {out_json}")

    # provenance lets inspect_cross_class() separate genuine label conflicts
    # (linked by exact hashes) from embedding-only cosine false positives.
    uf.provenance = {"uf_exact": uf_exact, "emb_edges": emb_edge_list}
    return report, uf, imgs


# --------------------------------------------------------------------------
# 6. Group-aware split (keeps duplicate clusters together)
# --------------------------------------------------------------------------
def grouped_split(uf, imgs, seed=42, test_frac=0.2):
    """
    Assign whole duplicate-clusters to train or test, stratified by class,
    so no cluster straddles the split. Returns (train_imgs, test_imgs).
    """
    rng = random.Random(seed)
    n = len(imgs)
    # cluster id per image
    cid = [uf.find(i) for i in range(n)]
    # group cluster -> its (majority) class + members
    from collections import defaultdict, Counter
    members = defaultdict(list)
    for i in range(n):
        members[cid[i]].append(i)
    # stratify clusters by their majority class
    by_class = defaultdict(list)
    for c, mem in members.items():
        maj = Counter(imgs[i]["class"] for i in mem).most_common(1)[0][0]
        by_class[maj].append((c, mem))

    test_flag = [False] * n
    for cls, clusts in by_class.items():
        rng.shuffle(clusts)
        total = sum(len(m) for _, m in clusts)
        target = test_frac * total
        acc = 0
        for _, mem in clusts:
            if acc < target:
                for i in mem:
                    test_flag[i] = True
                acc += len(mem)
    train = [imgs[i] for i in range(n) if not test_flag[i]]
    test = [imgs[i] for i in range(n) if test_flag[i]]
    print(f"grouped_split: train={len(train)}  test={len(test)} "
          f"({100*len(test)/n:.1f}% test)  |  0 clusters straddle the split")
    return train, test


# --------------------------------------------------------------------------
# 7. Optional helpers
# --------------------------------------------------------------------------
def _embedding_pass(imgs, uf, cos_thresh):
    """Uses dino_v1 / tf_dv1 / device from the caller's globals if available."""
    import builtins, torchvision.transforms as T
    g = globals()
    dino = g.get("dino_v1"); tf = g.get("tf_dv1"); dev = g.get("device")
    if dino is None:
        try:
            import __main__ as m
            dino = getattr(m, "dino_v1", None); tf = getattr(m, "tf_dv1", None)
            dev = getattr(m, "device", None)
        except Exception:
            pass
    if dino is None or tf is None:
        print("  [embedding pass skipped: dino_v1/tf_dv1 not found]")
        return 0, []
    import torch
    feats = []
    with torch.no_grad():
        for rec in imgs:
            try:
                im = Image.open(rec["path"]).convert("RGB")
                f = dino(tf(im).unsqueeze(0).to(dev)).cpu().numpy()[0]
            except Exception:
                f = np.zeros(384, dtype=np.float32)
            feats.append(f)
    F = np.array(feats, dtype=np.float32)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    n = len(F); edges = 0; edge_list = []; chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        sim = F[i0:i1] @ F.T
        rows, cols = np.where(sim >= cos_thresh)
        for r, c in zip(rows, cols):
            gi = i0 + r
            if gi < c:
                uf.union(gi, int(c)); edges += 1
                edge_list.append((int(gi), int(c)))
    print(f"  embedding pass: {edges} suspect pairs (cos>={cos_thresh})")
    return edges, edge_list


def _save_montages(multi, imgs, out_dir, k):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    big = sorted(multi.values(), key=len, reverse=True)[:k]
    for ci, g in enumerate(big):
        m = min(len(g), 6)
        fig, ax = plt.subplots(1, m, figsize=(3 * m, 3))
        if m == 1:
            ax = [ax]
        for j in range(m):
            ax[j].imshow(Image.open(imgs[g[j]]["path"]).convert("RGB"))
            ax[j].set_title(imgs[g[j]]["class"], fontsize=8)
            ax[j].axis("off")
        fig.suptitle(f"Cluster {ci} (size {len(g)})", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"dup_cluster_{ci:02d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {len(big)} montages -> {out_dir}")


# --------------------------------------------------------------------------
# 8. Cross-class inspection: genuine label conflicts vs embedding false positives
# --------------------------------------------------------------------------
def inspect_cross_class(uf, imgs, save_dir=None, out_json=None, verbose=True,
                        max_print=40):
    """
    A cross-class cluster is a duplicate group whose members carry >1 class label.
    Two very different causes, handled oppositely:
      EXACT-DUP  : an exact hash/md5 duplicate that sits under two class folders
                   -> a REAL annotation conflict; the true label is ambiguous.
      EMBEDDING  : distinct leaves linked only by the cosine pass
                   -> a false positive; not a label error.
    Uses uf.provenance['uf_exact'] to tell them apart.
    """
    n = len(imgs)
    prov = getattr(uf, "provenance", None)
    uf_exact = prov["uf_exact"] if prov else None

    full = {}
    for i in range(n):
        full.setdefault(uf.find(i), []).append(i)

    label_conflicts, suspect = [], []
    for g in full.values():
        if len(g) < 2:
            continue
        classes = {imgs[i]["class"] for i in g}
        if len(classes) < 2:
            continue
        if uf_exact is not None:
            sub = {}
            for i in g:
                sub.setdefault(uf_exact.find(i), []).append(i)
            conflict_subs = [s for s in sub.values()
                             if len({imgs[k]["class"] for k in s}) > 1]
        else:
            conflict_subs = [g]
        entry = {"classes": sorted(classes), "size": len(g),
                 "paths": [imgs[i]["path"] for i in g]}
        if conflict_subs:
            entry["conflict_paths"] = [imgs[k]["path"]
                                       for s in conflict_subs for k in s]
            label_conflicts.append(entry)
        else:
            suspect.append(entry)

    if verbose:
        print("=" * 60)
        print("CROSS-CLASS CLUSTER INSPECTION")
        print("=" * 60)
        print(f"GENUINE label conflicts (exact dup, differing labels): "
              f"{len(label_conflicts)}")
        print(f"SUSPECT (embedding-only links, likely false positives): "
              f"{len(suspect)}")
        print("-" * 60)
        for e in label_conflicts[:max_print]:
            print(f"  [CONFLICT] {' vs '.join(e['classes'])}  (size {e['size']})")
            for p in e.get("conflict_paths", e["paths"])[:4]:
                print(f"      {os.path.relpath(p)}")
        if suspect:
            print("-" * 60)
            print("  suspect examples (verify in montages; usually NOT errors):")
            for e in suspect[:10]:
                print(f"    [suspect] {' vs '.join(e['classes'])}  (size {e['size']})")

    result = {"n_conflicts": len(label_conflicts), "n_suspect": len(suspect),
              "label_conflicts": label_conflicts, "suspect": suspect}
    if out_json:
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {out_json}")
    if save_dir:
        _save_conflict_montages(label_conflicts, imgs, save_dir)
    return result


def _save_conflict_montages(conflicts, imgs, out_dir, k=20):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for ci, e in enumerate(conflicts[:k]):
        paths = e.get("conflict_paths", e["paths"])[:6]
        m = len(paths)
        fig, ax = plt.subplots(1, m, figsize=(3 * m, 3.2))
        if m == 1:
            ax = [ax]
        for j, p in enumerate(paths):
            try:
                ax[j].imshow(Image.open(p).convert("RGB"))
            except Exception:
                pass
            cls = os.path.basename(os.path.dirname(p))
            ax[j].set_title(cls, fontsize=8)
            ax[j].axis("off")
        fig.suptitle(f"CONFLICT {ci}: {' vs '.join(e['classes'])}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"conflict_{ci:02d}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {min(len(conflicts), k)} conflict montages -> {out_dir}")


def flagged_to_drop(uf, imgs, drop_conflicts=True, dedup_same_class=True):
    """
    Return a set of id(img_dict) to exclude for a clean run.
      drop_conflicts   : drop every image in a genuine cross-class EXACT cluster
                         (ambiguous label -> unsafe to keep either copy).
      dedup_same_class : keep only one representative per same-class exact cluster.
    Embedding-only links are NOT used here, so cosine false positives never cause
    drops. Because grouped_split keeps whole clusters on one side, filtering by id
    after the split stays leakage-safe.
    """
    n = len(imgs)
    prov = getattr(uf, "provenance", None)
    uf_exact = prov["uf_exact"] if prov else uf
    drop = set()

    sub = {}
    for i in range(n):
        sub.setdefault(uf_exact.find(i), []).append(i)

    n_conf, n_dedup = 0, 0
    for s in sub.values():
        if len(s) < 2:
            continue
        cls = {imgs[k]["class"] for k in s}
        if drop_conflicts and len(cls) > 1:
            for k in s:
                drop.add(id(imgs[k])); n_conf += 1
        elif dedup_same_class:
            for k in s[1:]:
                drop.add(id(imgs[k])); n_dedup += 1

    print(f"flagged_to_drop: {len(drop)} images "
          f"({n_conf} from label conflicts, {n_dedup} redundant same-class dups)")
    return drop
