# Label-efficient tea leaf disease screening — code, artifacts, and deployed model

Code and released artifacts for

> M. M. H. Munna, A. H. Kabir, R. F. Tulona, M. M. U. Mulk, "Label-Efficient Tea Leaf Disease Screening for
> Offline Smartphone Deployment: Exposing and Controlling Capture Bias," manuscript under revision, IEEE Access.

Dataset: **TeaLeafDiseaseBD**, Mendeley Data, doi:[10.17632/jcvxm4gh7d.3](https://doi.org/10.17632/jcvxm4gh7d.3)
(13,084 images, 7 classes, eleven gardens in Sylhet, Moulvibazar and Habiganj, Bangladesh).

## What is in this repository

| Path | Content |
|---|---|
| `models/tea_screener.onnx` | deployed binary screener: MobileNetV3-Small, 1.52 M parameters, 6.09 MB, ONNX opset 13, dynamic batch |
| `artifacts/` | split, frozen features, per-image student scores, and every result JSON (see `artifacts/README.md`) |
| `analysis/reconcile.py` | regenerates the in-distribution numbers (student, probe, label curve, triage, Figs. 7–8) from `artifacts/` |
| `analysis/recalibrate.py` | external label-free threshold and few-label recalibration, near-duplicate groups held out |
| `notebooks/revision_v1_reproduce.ipynb` | rebuilds the 70-label pipeline from the artifacts, trains the student, exports and verifies the ONNX model |
| `notebooks/external_test.ipynb` | external evaluation on two independently collected tea datasets (GPU, images needed) |
| `deploy/benchmark_onnx.py` | latency, memory, and approximate energy on any device, including Android/Termux |
| `scripts/original/`, `notebooks/original/` | the original experiment code, copied verbatim (see `scripts/original/MANIFEST.md`) |
| `tools/verify.py` | checksums + regeneration of every image-free number, with assertions |

## Quick start (CPU, about one minute, no images)

```bash
pip install numpy scipy scikit-learn matplotlib onnxruntime
python tools/verify.py          # checksums, regenerate, assert headline numbers
make bench                      # latency + memory of the deployed model on this machine
```

## Method in one paragraph

Each leaf is segmented (Otsu on HSV value and saturation, largest component, holes filled), cropped to its
bounding box with 6% padding and resized to 224×224, which removes leaf scale from the input. Frozen
DINO ViT-S/16 [CLS] features (ℓ2-normalised) feed a multinomial logistic probe (C = 1.0) trained on **10
expert-labelled leaves per class (70 labels, ≈9 expert-minutes)**. The probe pseudo-labels the remaining
training images (hard labels, threshold at the healthy-prevalence quantile), and an ImageNet-initialised
MobileNetV3-Small student is trained on them and exported to ONNX. The encoder never ships.

## Results and where each number comes from

Test split: 3,929 images (86.4% diseased), group-aware so that no near-duplicate cluster crosses train/test.
Mean ± SD over five independent draws of the 70-label seed set.

| Result | Value | Source | Regenerate |
|---|---|---|---|
| Distilled student (revision_v1, 5 seeds), binary AUC | 0.985 ± 0.006 | `E_scores_5rep.npz` | `tools/verify.py` |
| Student accuracy / sensitivity / specificity (threshold 0.5) | 96.6% / 98.2% / 86.2% | same | `tools/verify.py` |
| **Exported model `tea_screener.onnx`** (seed draw 1) | AUC 0.986; accuracy 96.5%; sensitivity 98.4%; specificity 84.3% (threshold 0.5) | `export_report.json` | `notebooks/revision_v1_reproduce.ipynb` |
| Frozen probe (teacher) AUC; student − teacher | 0.985 ± 0.004; +0.0001 | `features_dinov1_crop.npz` | `tools/verify.py` |
| Labels-only MobileNetV3 at 70 labels | 0.957 ± 0.037 (paired p = 0.13; variance ratio 6.2×, Pitman–Morgan p = 0.004) | `E_distillation_5rep.json` | `tools/verify.py` |
| 7-class probe accuracy at 70 labels | 0.685 ± 0.018 | features | `tools/verify.py` |
| Triage at 10% field prevalence, 95% sensitivity | 84.7% of leaves cleared, NPV 0.994 | student scores | `tools/verify.py` |
| External zero-shot AUC, probe (EWU / Allynugger) | 0.874 ± 0.029 / 0.915 ± 0.016 | `external_results_v3.json` | `notebooks/external_test.ipynb` |
| External specificity at the in-distribution threshold | probe 0.32 / 0.53; deployed 0.62 / 0.71 | `external_results_v3.json` | `notebooks/external_test.ipynb` |
| Deployed model, external zero-shot AUC (EWU / Allynugger) | 0.883 / 0.889 | `external_results_v3.json` | `notebooks/external_test.ipynb` |
| EWU after 10 / 20 local labels per class (40 draws, near-duplicates held out) | AUC 0.973 ± 0.009 / 0.983 ± 0.005 | `external_results_v3.json` | `notebooks/external_test.ipynb` |
| Capture-bias controls, shape baselines, clutter, mask perturbation | Tables 5, 7, 8, 13 | original scripts | `scripts/original/` |
| On-device latency (MediaTek Helio G92) | 26.9 ms single-thread (p95 28.5), 17.8 ms all cores (p95 20.1); +28.5 MB memory | `device_benchmark_helio_g92.json` | `deploy/benchmark_onnx.py` |

## Using the deployed model

Input: RGB images `[N, 3, 224, 224]`, the leaf crop described above, scaled to
[0, 1] and normalised with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).
Output: `logits [N, 2]`; softmax index 1 is the probability that the leaf is diseased. The batch dimension is dynamic.

Provenance: trained by `notebooks/revision_v1_reproduce.ipynb` on seed draw 1 of the published run, using the cached
features and the original segmentation modules. Its pseudo-labels reproduce the published run exactly
(τ = 0.7872, pseudo-label accuracy 0.9608); see `artifacts/export_report.json` for checks and SHA-256.

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('models/tea_screener.onnx')
z = sess.run(None, {'input': x})[0][0]            # x: float32 [N, 3, 224, 224], preprocessed as above
p_diseased = np.exp(z[1]) / np.exp(z).sum()
```

**Intended use.** A triage aid that decides which leaves an agronomist should examine. It is not a
diagnostic system and does not replace expert inspection or treatment decisions.

**Limitations.**
- Trained on detached single leaves photographed indoors on a dark cloth backdrop. The Otsu segmentation
  depends on that backdrop and fails on light or textured backgrounds.
- On two independently collected datasets with light backdrops, disease ranking transfers (AUC 0.87–0.92)
  but the in-distribution threshold does not: specificity falls to 0.32–0.53. A new site should either
  follow the dark-backdrop capture protocol or recalibrate with about ten labelled leaves per class.
- Not validated on in-canopy photographs or prospectively in a plantation. Garden identifiers were not
  retained, so estate-held-out evaluation within TeaLeafDiseaseBD is not possible.
- Seven-class identification (0.685 at 70 labels) is not adequate for unsupervised treatment decisions.

## Citation

See `CITATION.cff`. Please also cite the TeaLeafDiseaseBD dataset and, for the external evaluation, the
original EWU and Allynugger datasets.

## Licence

Code: MIT (see `LICENSE`). The dataset is distributed under its own licence on Mendeley Data.
External-dataset features in `artifacts/` are derived from third-party data and remain subject to those
datasets' licences.
