# Released artifacts (revision_v1)

| File | Content | Produced by |
|---|---|---|
| `_split.json` | group-aware 70/30 split (seed 42): 3,929 test files as `<class>/<file>`, 9,155 train | `scripts/original/leakage_audit.py` (`phash=0, dhash=0`) |
| `features_dinov1_crop.npz` | frozen DINO ViT-S/16 [CLS] features (ℓ2-normalised) of leaf-crop-normalised images: `Xtr, ytr, dtr, Xte, yte, dte` | revision_v1 notebook, crop cache (bbox + 6% pad, 224², JPEG q95) |
| `E_distillation_5rep.json` | 5 seed draws × {labels-only, probe+distillation}: per-seed AUCs, taus, pseudo-label accuracy, statistics | revision_v1 notebook, Experiment E |
| `E_scores_5rep.npz` | per-image test scores of the 10 students above, in `_split.json` order | revision_v1 notebook, Experiment E |
| `reconciled_numbers.json` | every manuscript number derivable without images (Tables 9–11, Figs. 7–8) | `analysis/reconcile.py` |
| `feat_EWU_box.npz` (3,007 rows), `feat_Ally_whole.npz` (883) | DINO features of the two external datasets (crops as in `notebooks/external_test.ipynb`) | `notebooks/external_test.ipynb` |
| `external_results_v3.json` | final external evaluation (Table 15): 3,007 + 883 images, probe and deployed model, label-free threshold, 40-draw recalibration | `notebooks/external_test.ipynb` |
| `regen_tables_8_9.json` | Table 8 (synthetic clutter, final split, five probes) and the Table 9 shape-control column | `notebooks/regenerate_tables_8_9.ipynb` |
| `external_blind_transfer.json` | Table 17: field-adapted students S0/S1/S2 (three seeds) scored on External datasets 1 and 2 | `notebooks/external_blind_transfer.ipynb` |
| `cross_dataset_confound.json` | §IV-F, Fig. 5: area ratio, shape-only AUC and attention collapse on five datasets | `scripts/original/cross_dataset_confound.py` |
| `device_benchmark_unisoc_t606.json` | on-device latency and memory on a second handset (itel S665L, Unisoc T606) | `deploy/benchmark_onnx.py` |
| `device_benchmark_helio_g92.json` | on-device latency and memory of `models/tea_screener.onnx` on a Helio G92 handset | `deploy/benchmark_onnx.py` |
| `feat_Field_whole.npz`, `feat_Field_sam.npz` | DINO features of the 2,500 field photographs (whole frame; SAM ViT-B centre-point leaf crop): `X, y, area, cls` | `notebooks/field_test.ipynb` |
| `field_results.json` | zero-shot field evaluation: probe and deployed model, four-class view, label-free threshold, recalibration curve | `notebooks/field_test.ipynb` |
| `experiment_F_results.json` | first field-adaptation run (seed 1) | `notebooks/experiment_F_field_student.ipynb` |
| `experiment_F_robustness.json` | field adaptation S0/S1/S2 over three seeds, prevalence 16/26/36%, paired gains (Table 16) | `notebooks/experiment_F_robustness.ipynb` |
| `experiment_F_scores.npz` | per-image scores of every Experiment F student on the held-out field half (1,248) and our test split, with field class and near-duplicate group | `notebooks/experiment_F_robustness.ipynb` |
| `export_report_field_S1.json` | export check and SHA-256 of the field-adapted S1 student (seed 1), `tea_screener_field_S1_rep1.onnx` | `notebooks/experiment_F_robustness.ipynb` |
| `recal_dup0.95.json`, `recal_dup0.90.json` | label-free threshold and few-label recalibration, near-duplicate groups held out | `analysis/recalibrate.py` |

Labels: class index = alphabetical folder order (Algal Leaf Spot 0, Brown Blight 1, Gray Blight 2, Healthy 3,
Helopeltis 4, Red Rust 5, Red Spider 6); `d* = 1` means diseased.

Field data: Chowdhury & Arman (2025), Mendeley Data, doi:10.17632/3x42rbj8yv.1, CC BY 4.0. Field labels are
the dataset's four classes (Blight, Healthy, Helopeltis, Red Rust) in `cls`.

The external- and field-dataset features are derived from third-party data; check each dataset's licence before
redistributing them, and cite the original datasets.
