# Label-efficient tea leaf disease screening — code, artifacts, and deployed model

Repository: https://github.com/munna237/Tealeaf-screener

Code and released artifacts for

> M. M. H. Munna, A. H. Kabir, R. F. Tulona, M. M. U. Mulk, "Label-Efficient Tea Leaf Disease Screening for
> Offline Smartphone Deployment: Exposing and Controlling Capture Bias," manuscript under revision.

Dataset: **TeaLeafDiseaseBD**, Mendeley Data, doi:[10.17632/jcvxm4gh7d.3](https://doi.org/10.17632/jcvxm4gh7d.3)
(13,084 images, 7 classes, eleven gardens in Sylhet, Moulvibazar and Habiganj, Bangladesh).

## What is in this repository

| Path | Content |
|---|---|
| `models/tea_screener_field_S1.onnx` | field-adapted student S1 (seed 1) of Section IV-Q, same architecture and size; exported in the notebook as `tea_screener_field_S1_rep1.onnx` and renamed here; SHA-256 in `artifacts/export_report_field_S1.json` |
| `models/tea_screener.onnx` | deployed binary screener: MobileNetV3-Small, 1.52 M parameters, 6.09 MB, ONNX opset 13, dynamic batch |
| `artifacts/` | split, frozen features, per-image student scores, and every result JSON (see `artifacts/README.md`) |
| `analysis/reconcile.py` | regenerates the in-distribution numbers (student, probe, label curve, triage, Figs. 7–8) from `artifacts/` |
| `analysis/recalibrate.py` | external label-free threshold and few-label recalibration, near-duplicate groups held out |
| `notebooks/revision_v1_reproduce.ipynb` | rebuilds the 70-label pipeline from the artifacts, trains the student, exports and verifies the ONNX model |
| `notebooks/external_test.ipynb` | external evaluation on two independently collected tea datasets (GPU, images needed) |
| `notebooks/field_test.ipynb` | zero-shot evaluation on 2,500 field photographs from four Habiganj estates: whole frame, centre crop, SAM leaf crop (GPU, images needed) |
| `notebooks/experiment_F_field_student.ipynb`, `notebooks/experiment_F_robustness.ipynb` | zero-label field adaptation: re-distil the student on probe-labelled field photographs (S0/S1/S2, seeds 0–2, prevalence sensitivity); seeds 3–9 in `notebooks/more_seeds.ipynb` |
| `notebooks/external_blind_transfer.ipynb` | scores the field-adapted students (S0/S1/S2, seeds 0–2) on External datasets 1 and 2 (images needed, inference only); seeds 3–9 in `notebooks/more_seeds.ipynb` |
| `notebooks/regenerate_tables_8_9.ipynb` | rebuilds the shape-control column of Table 10 on the final split with the published probes (its Table 9 run is superseded by `regenerate_tables_2_8_14.ipynb`; images needed) |
| `analysis/fig_fewshot.py` | redraws Fig. 6 (label efficiency with the shape control) from `reconciled_numbers.json` and `regen_tables_8_9.json` |
| `analysis/probe_C_cv.py` | seven-class accuracy at C = 1 / 10 / 100 and with C chosen by 5-fold CV inside the 70 labels (CPU, cached features) |
| `notebooks/regenerate_tables_2_8_14.ipynb` | rebuilds Tables 2, 9 and 15 with the crop recipe that reproduces the cached features; times feature extraction (images needed) |
| `analysis/fig_qualitative.py` | composes Fig. 9 (our capture vs the external and field photographs, Otsu vs SAM) from Fig. 1 and the contact sheets |
| `analysis/local_threshold.py` | local calibration: sets a site's flag threshold from n labelled diseased leaves (order-statistic rule) and reports sensitivity, specificity and the cleared share over 2,000 resamples (CPU, released scores) |
| `notebooks/more_seeds.ipynb` | extends Experiment F to 10 seeds and Experiment E to 10 draws, and repeats the blind transfer for the new seeds (GPU, resumable) |
| `notebooks/controls_70labels.ipynb` | shape residualisation and C3 (shape-matched pairs) with the five published 70-label probes (CPU, images needed for the masks) |
| `notebooks/sam_confound.ipynb` | the four-dataset confound diagnostic with Segment Anything masks, with a pre-declared positive control on TeaLeafDiseaseBD (GPU, images needed) |
| `notebooks/sam_confound_followup.ipynb` | reruns the SAM diagnostic with the same sample and masks, keeps every per-image value, and adds class-blind mask QC, median area ratio, within-species shape AUC and the Algorithm 2 verdict with C3 (GPU, ~2 h; later cells CPU from the .npz) |
| `notebooks/baseline_tuned.ipynb`, `notebooks/baseline_transfer.ipynb` | tuned supervised fine-tunes and ImageNet linear probes on the ten label draws of Table 12, and their transfer to the field and external photographs (GPU, images needed) |
| `analysis/tuned_baselines.py` | paired tests for those baselines (Table 12, Table 2 note, Tables 17–18), from the released results, no images |
| `analysis/ten_draws.py` | regenerates the ten-draw / ten-seed numbers (Tables 12, 17, 18) and Table 6 from released scores and per-image SAM values, no images |
| `analysis/field.py` | regenerates the field numbers (Section IV-Q, Table 17) from released scores and features, no images |
| `deploy/benchmark_onnx.py` | latency, memory, and approximate energy on any device, including Android/Termux |
| `notebooks/original/`, `scripts/original/` | the original study code (19 notebooks, 17 scripts), copied verbatim and mapped to the paper in `scripts/original/MANIFEST.md` |
| `tools/verify.py` | checksums + regeneration of every image-free number, with assertions |

## Quick start (CPU, about one minute, no images)

```bash
pip install numpy scipy scikit-learn matplotlib onnxruntime
python tools/verify.py          # checksums, regenerate, assert every number marked ✓ below
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

Numbers as printed in the revised manuscript, in the order of its tables. Test split: 3,929 images (86.4% diseased),
group-aware so that no near-duplicate cluster crosses train/test. *Label draws* are the random choices of the 70
labelled leaves; unless a row says otherwise, probe results use draws 0–4 (the same five probes throughout), and
each student is paired with one label draw and one training seed. `python tools/verify.py` asserts every row
marked ✓ without images.

| Paper | Result | Value | Source (`artifacts/`) | Regenerate |
|---|---|---|---|---|
| Abstract, Table 12 | Distilled student, binary AUC, **ten label draws** ✓ | **0.978 ± 0.014**; 95.8% accuracy, 97.8% sensitivity, 83.0% specificity (balanced 90.4%) | `E_scores_5rep.npz` + `E_scores_rep5_9.npz` | `analysis/ten_draws.py` |
| Table 12, Sec. IV-M | Labels-only MobileNetV3, student recipe (Adam, lr 1e-3, 30 epochs), ten draws; paired tests ✓ | 0.956 ± 0.026; gain +0.022, paired t p = 0.022, Wilcoxon p = 0.004; SD ratio 1.9×, Pitman–Morgan p = 0.075 (not significant) | same | `analysis/ten_draws.py` |
| Table 12 | Frozen probe (teacher), ten draws ✓ | 0.980 ± 0.010; student − teacher −0.002 on average | `features_dinov1_crop.npz` | `analysis/ten_draws.py` |
| Table 12, Sec. IV-M | Tuned supervised fine-tunes (AdamW, 150 epochs, strong aug.; lr 1e-4 / 3e-4), same ten draws ✓ | 0.975 ± 0.008 / 0.976 ± 0.012; vs probe + distillation +0.003 / +0.002, paired t p = 0.53 / 0.65, Wilcoxon p = 0.62 / 1.0; Pitman–Morgan p = 0.16 / 0.65 | `baseline_tuned.json`, `baseline_tuned_scores.npz` | `notebooks/baseline_tuned.ipynb` (GPU); `analysis/tuned_baselines.py` |
| Table 12, Table 2 note | Linear probes on frozen ImageNet MobileNetV3-Small / ResNet-50, same ten draws ✓ | 0.960 ± 0.022 / 0.961 ± 0.006; DINO probe ahead by 0.021 / 0.020 on 10/10 draws (t p = 0.005 / <0.001, Wilcoxon p = 0.002; survives Bonferroni at 0.0125) | same | same |
| Tables 17, 18, Sec. IV-Q | Tuned fine-tunes on the field and external datasets ✓ | field held-out half 0.857 / 0.836 (whole), 0.916 / 0.910 (centre); S1 better on 10/10 draws (Wilcoxon p = 0.002); External 1 0.890 / 0.912, External 2 0.861 / 0.905, no significant difference from S1 | `baseline_transfer.json`, `baseline_transfer_scores.npz` | `notebooks/baseline_transfer.ipynb` (GPU, images needed); `analysis/tuned_baselines.py` |
| Sec. IV-J, IV-M | Same quantities over the first five draws ✓ | student 0.985 ± 0.006 (96.6% acc.); labels-only 0.957 ± 0.037 (p = 0.13; 6.2×, p = 0.004); probe 0.985 ± 0.004 | `E_scores_5rep.npz` | `analysis/reconcile.py` |
| Sec. IV-N | Exported model `tea_screener.onnx` (draw 1) ✓ | AUC 0.986; accuracy 96.5%; sensitivity 98.4%; specificity 84.3% | `export_report.json` | `notebooks/revision_v1_reproduce.ipynb` |
| Table 2 | Encoders, 70-label probe ✓ | DINOv1 0.985, CLIP 0.986, DINOv2 0.979 | `regen_tables_2_8_14.json` | `notebooks/regenerate_tables_2_8_14.ipynb` |
| Table 5, Fig. 5 | Zero-annotation pipeline and its confound (Otsu) | 0.905; shape-only 0.834; within-leaf 0.543 | `cross_dataset_confound.json` | `scripts/original/` |
| Table 6 | Confound diagnostic with Segment Anything ✓ | ours: shape 0.904, collapse +0.338, area ratio 1.28 (median 1.68, class-blind QC 1.57), *Confound present*; PlantVillage 0.768 / PlantDoc 0.753 *Inconclusive*; RoCoLe 0.725 / RiceLeafBD 0.615 *Pathology* | `sam_confound_per_image.npz`, `sam_confound_followup.json` | `analysis/ten_draws.py`; `notebooks/sam_confound_followup.ipynb` |
| Table 8 | Shape controls, all labels / 70 labels ✓ (70-label column) | residualised 0.996 / 0.971; matched pairs 0.999 / 0.970 (shape on pairs 0.618 / 0.510, 510 pairs) | `controls_70labels.json` | `notebooks/controls_70labels.ipynb` |
| Table 9 | Simulated clutter, α = 0 / 0.2 / 0.4 / 0.6 ✓ | 0.985 / 0.979 / 0.962 / 0.944 | `regen_tables_2_8_14.json` | `notebooks/regenerate_tables_2_8_14.ipynb` |
| Table 10, Fig. 6 | Label efficiency at 70 labels; shape control ✓ | probe 0.985 ± 0.004, 7-class 0.685 ± 0.018; shape 0.725 ± 0.029 (all labels 0.792) | `reconciled_numbers.json`, `regen_tables_8_9.json` | `analysis/reconcile.py` |
| Sec. IV-J | 7-class accuracy vs C ✓ | 0.685 / 0.743 / 0.752 at C = 1 / 10 / 100; C by CV 0.743 | `probe_C_cv.json` | `analysis/probe_C_cv.py` |
| Sec. IV-N | Seed set drawn without class stratification (70 random leaves) ✓ | AUC 0.981 ± 0.010 (100 draws; 5th pct 0.964) vs stratified 0.980 ± 0.010; 7-class 0.51 (0.69 with balanced class weights) | `features_dinov1_crop.npz` | `analysis/random_seed_set.py` |
| Table 11 | Triage at 10% prevalence, 95% sensitivity ✓ | 84.7% of leaves cleared, NPV 0.994 | student scores | `analysis/reconcile.py` |
| Tables 13, 14 | On-device, Helio 55 (MediaTek Helio G92) / itel S665L (Unisoc T606) | 26.9 / 32.8 ms per 224×224 crop (one thread; segmentation not included); +28.5 / +29.4 MB; 36.9 ± 1.4 / 28.3 ± 1.5 mJ per leaf | `device_benchmark_*.json` | `deploy/benchmark_onnx.py`, `deploy/energy_adb.py` |
| Table 15 | Mask perturbation ✓ | clean 0.985; shift 0.976; jitter 0.980 | `regen_tables_2_8_14.json` | `notebooks/regenerate_tables_2_8_14.ipynb` |
| Sec. IV-P | Within capture day / within file prefix (IMG, PXL) | 0.978–0.995 / 0.972 and 0.995 | `reconciled_numbers.json` | `analysis/reconcile.py` |
| Table 16 | External datasets 1 / 2, zero-shot ✓ | probe 0.874 / 0.915; deployed 0.883 / 0.889 (specificity 0.62 / 0.71); 10 / 20 local labels per class on dataset 1: probe 0.973 / 0.983 | `external_results_v3.json`, `recal_dup0.95.json` | `analysis/recalibrate.py` |
| Table 17 | Field, zero-shot probe ✓ / deployed | 0.962 (whole) / 0.973 (centre) / 0.970 (SAM); deployed 0.639 | `field_results.json` | `analysis/field.py` |
| Table 17 | Field adaptation, **ten seeds**, whole frame ✓ | S0 0.727 ± 0.094, S1 0.974 ± 0.010, S2 0.988 ± 0.002; S1 > S0 on every seed (Wilcoxon p = 0.002); on our test split S1 0.978 vs S0 0.980 | `experiment_F_robustness.json` (seeds 0–2) + `experiment_F_seeds3_9.json` | `analysis/ten_draws.py` |
| Table 18 | Blind transfer of the field-adapted students, **ten seeds** ✓ | S1 0.899 ± 0.073 / 0.920 ± 0.028 vs S0 0.853 ± 0.099 / 0.886 ± 0.068; S1 > S0 on 9 / 7 of 10 seeds | `external_blind_transfer_10seeds.json` | `analysis/ten_draws.py` |
| Table 19 | Deployment decision table | collects Tables 12, 16 and 17 | — | — |
| Fig. 9 | Capture conditions, Otsu vs SAM | `figures/fig_qualitative.png` | contact sheets | `analysis/fig_qualitative.py` |

Superseded runs kept for provenance and **not** quoted in the paper: `experiment_F_robustness.json` summary and
`external_blind_transfer.json` (three seeds; replaced by the ten-seed rows above), `sam_confound.json` (first SAM
run, summary only), `local_threshold.py` (analysis not included in the paper).

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
- On 2,500 field photographs from four estates outside our collection (Chowdhury & Arman, 2025), the frozen
  probe ranks disease well (AUC 0.96–0.97) but this deployed model does not (0.64 whole frame). Re-distilling
  the student on probe-labelled local photographs (`notebooks/experiment_F_field_student.ipynb`) restores
  AUC 0.97 with no new labels. The adaptation and test halves share estates and season, so this is adaptation
  to a site, not blind transfer; each photo is framed on one leaf, so dense-canopy detection is untested.
- Not validated prospectively in a plantation. Garden identifiers were not retained, so estate-held-out
  evaluation within TeaLeafDiseaseBD is not possible.
- Seven-class identification (0.685 at 70 labels) is not adequate for unsupervised treatment decisions.

## Citation

See `CITATION.cff`. Please also cite the TeaLeafDiseaseBD dataset and, for the external evaluation, the
original EWU and Allynugger datasets and, for the field evaluation, Chowdhury & Arman (2025),
Tea Leaf Disease Image Dataset, Mendeley Data, doi:[10.17632/3x42rbj8yv.1](https://doi.org/10.17632/3x42rbj8yv.1).

## Licence

Code: MIT (see `LICENSE`). The dataset is distributed under its own licence on Mendeley Data.
External- and field-dataset features in `artifacts/` are derived from third-party data and remain subject to those
datasets' licences.
