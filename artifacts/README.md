# Released artifacts (revision_v1)

| File | Content | Produced by |
|---|---|---|
| `_split.json` | group-aware 70/30 split (seed 42): 3,929 test files as `<class>/<file>`, 9,155 train | `scripts/original/leakage_audit.py` (`phash=0, dhash=0`) |
| `features_dinov1_crop.npz` | frozen DINO ViT-S/16 [CLS] features (ℓ2-normalised) of leaf-crop-normalised images: `Xtr, ytr, dtr, Xte, yte, dte` | revision_v1 notebook, crop cache (bbox + 6% pad, 224², JPEG q95) |
| `E_distillation_5rep.json` | 5 seed draws × {labels-only, probe+distillation}: per-seed AUCs, taus, pseudo-label accuracy, statistics | revision_v1 notebook, Experiment E |
| `E_scores_5rep.npz` | per-image test scores of the 10 students above, in `_split.json` order | revision_v1 notebook, Experiment E |
| `reconciled_numbers.json` | every manuscript number derivable without images (Tables 10–12, Figs. 7–8) | `analysis/reconcile.py` |
| `feat_EWU_box.npz` (3,007 rows), `feat_Ally_whole.npz` (883) | DINO features of the two external datasets (crops as in `notebooks/external_test.ipynb`) | `notebooks/external_test.ipynb` |
| `external_results_v3.json` | final external evaluation (Table 16): 3,007 + 883 images, probe and deployed model, label-free threshold, 40-draw recalibration | `notebooks/external_test.ipynb` |
| `regen_tables_8_9.json` | Table 10 shape-control column (its Table 9 run is superseded by the file below) | `notebooks/regenerate_tables_8_9.ipynb` |
| `regen_tables_2_8_14.json` | Tables 2, 9 and 15 rebuilt with the crop recipe that reproduces the cached features; feature-extraction timing; dataset file listing | `notebooks/regenerate_tables_2_8_14.ipynb` |
| `probe_C_cv.json` | seven-class accuracy at C = 1 / 10 / 100 and with C chosen by CV inside the 70 labels | `analysis/probe_C_cv.py` |
| `external_blind_transfer.json` | seeds 0–2 of the blind transfer; its three-seed summary is superseded by `external_blind_transfer_10seeds.json` (Table 18) | `notebooks/external_blind_transfer.ipynb` |
| `cross_dataset_confound.json` | §IV-F, Fig. 5: area ratio, shape-only AUC and attention collapse on five datasets | `scripts/original/cross_dataset_confound.py` |
| `device_benchmark_unisoc_t606.json` | on-device latency and memory on a second handset (itel S665L, Unisoc T606) | `deploy/benchmark_onnx.py` |
| `device_benchmark_helio_g92.json` | on-device latency and memory of `models/tea_screener.onnx` on a Helio G92 handset | `deploy/benchmark_onnx.py` |
| `feat_Field_whole.npz`, `feat_Field_sam.npz` | DINO features of the 2,500 field photographs (whole frame; SAM ViT-B centre-point leaf crop): `X, y, area, cls` | `notebooks/field_test.ipynb` |
| `field_results.json` | zero-shot field evaluation: probe and deployed model, four-class view, label-free threshold, recalibration curve | `notebooks/field_test.ipynb` |
| `experiment_F_results.json` | first field-adaptation run (seed 1) | `notebooks/experiment_F_field_student.ipynb` |
| `experiment_F_robustness.json` | field adaptation S0/S1/S2 seeds 0–2 and the prevalence rows of Table 17 (16/26/36%); its three-seed summary is superseded by the ten-seed analysis (`analysis/ten_draws.py`) | `notebooks/experiment_F_robustness.ipynb` |
| `experiment_F_scores.npz` | per-image scores of every Experiment F student on the held-out field half (1,248) and our test split, with field class and near-duplicate group | `notebooks/experiment_F_robustness.ipynb` |
| `export_report_field_S1.json` | export check and SHA-256 of the field-adapted S1 student (seed 1), `tea_screener_field_S1_rep1.onnx` | `notebooks/experiment_F_robustness.ipynb` |
| `recal_dup0.95.json`, `recal_dup0.90.json` | label-free threshold and few-label recalibration, near-duplicate groups held out | `analysis/recalibrate.py` |

Labels: class index = alphabetical folder order (Algal Leaf Spot 0, Brown Blight 1, Gray Blight 2, Healthy 3,
Helopeltis 4, Red Rust 5, Red Spider 6); `d* = 1` means diseased.

Field data: Chowdhury & Arman (2025), Mendeley Data, doi:10.17632/3x42rbj8yv.1, CC BY 4.0. Field labels are
the dataset's four classes (Blight, Healthy, Helopeltis, Red Rust) in `cls`.

The external- and field-dataset features are derived from third-party data; check each dataset's licence before
redistributing them, and cite the original datasets.
| `E_distillation_10rep.json`, `E_scores_rep5_9.npz` | Experiment E draws 5–9 (labels-only and probe + distillation students) and the ten-draw statistics (Table 12) | `notebooks/more_seeds.ipynb` |
| `experiment_F_seeds3_9.json`, `experiment_F_scores_seeds3_9.npz`, `experiment_F_summary_10seeds.json` | field adaptation S0/S1/S2 seeds 3–9 and the ten-seed summary (Table 17) | `notebooks/more_seeds.ipynb` |
| `external_blind_transfer_10seeds.json` | blind transfer of the thirty field-adapted students to External datasets 1 and 2 (Table 18) | `notebooks/more_seeds.ipynb` |
| `controls_70labels.json` | C2 and C3 with the five 70-label probes (Table 8, 70-label column) | `notebooks/controls_70labels.ipynb` |
| `sam_confound.json` | first Segment Anything run of the cross-dataset diagnostic (summary only) | `notebooks/sam_confound.ipynb` |
| `sam_confound_followup.json`, `sam_confound_per_image.npz` | Table 6: same sample and masks, per-image values, QC, species, Algorithm 2 verdicts; contact sheets in `figures/sam_contact/` | `notebooks/sam_confound_followup.ipynb` |
