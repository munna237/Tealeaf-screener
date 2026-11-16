# Original study code

Copied **verbatim** from the authors' Google Drive (notebook outputs stripped; files renamed for order only — the
original name is listed). Nothing here was rewritten. Code added during the revision lives in `analysis/`,
`deploy/`, `tools/` and the two top-level notebooks.

## Notebooks (`notebooks/original/`)

| File | Original name | Paper element |
|---|---|---|
| `01_zero_annotation_attention_pipeline.ipynb` | Phase 1 notebook for the tea leaf DINOv2 attention analysis.ipynb | §III-D, §IV-B: frozen DINO attention pipeline, UAHS head selection, external-dataset downloads used in §IV-F |
| `02_multi_seed_runs.ipynb` | Multi seed.ipynb | §IV-B: multi-seed runs of the zero-annotation pipeline |
| `03_integrity_audit.ipynb` | Audit.ipynb | §III-B, §IV-A: perceptual-hash duplicate audit (uses leakage_audit.py) |
| `04_leakage_audit_and_triage.ipynb` | Untitled38.ipynb | §IV-A group-aware split; original Table 10 triage (uses leakage_audit.py, triage_analysis.py) |
| `05_localisation_eval.ipynb` | Untitled37.ipynb | §IV-D, Table 3, Fig. 3: localisation vs trivial baselines (uses localization_eval.py, annotate_tool.py) |
| `06_localisation_validation.ipynb` | tealeaf_localisation_validation.ipynb | §IV-D, Table 3: centre-bias and dataset-prior baselines |
| `07_leaf_scale_confound.ipynb` | Untitled42.ipynb | §IV-E, Table 5, Fig. 4: capture-bias controls C1–C3 (uses shape_confound.py, feature_shape_control.py) |
| `08_confound_clustering_label_curve.ipynb` | Tea Leaf Re .ipynb | §IV-F Fig. 5 (cross_dataset_confound.py); §IV-G Table 6 (overcluster_pipeline.py, select_k_and_stability.py); §IV-J (few_shot_curve.py) |
| `09_multiclass_clustering.ipynb` | Untitled39.ipynb | §IV-G: clustering under label-free model selection (uses multiclass_cluster.py) |
| `10_deployment_clutter_student.ipynb` | Tea Deployment.ipynb | §IV-I Table 8 (clutter_robustness*.py); earlier student runs (few_shot_student*.py); fig_diagnostics.py |
| `11_revision_experiments.ipynb` | tealeaf_revision_experiments (1).ipynb | §IV-C Table 2 (encoder comparison), §IV-O Table 13 (mask perturbation), crop cache |
| `11b_revision_experiments_earlier_version.ipynb` | tealeaf_revision_experiments.ipynb | earlier version of 11 (kept for completeness) |
| `12_experiment_E_five_seed_distillation.ipynb` | tealeaf_experimentE_5rep.ipynb | §IV-M Table 11; produces E_distillation_5rep.json and E_scores_5rep.npz |
| `13_figures_results.ipynb` | Tea leaf figure.ipynb | Figs. 6–8 (original versions; uses make_all_figures.py) |
| `14_figures_qualitative.ipynb` | TEA LEAF FIGURES.ipynb | Figs. 1, 3 and qualitative panels (uses make_qualitative_figures.py, make_results_figures.py) |
| `15_dinov2_attention_hooks.ipynb` | Untitled23.ipynb | attention extraction for DINOv2 (encoder comparison, §IV-C) | *(role inferred from content — please confirm)*
| `16_feature_tsne_umap.ipynb` | Untitled32.ipynb | t-SNE/UMAP of DINOv1 features (exploratory; not a reported figure) | *(role inferred from content — please confirm)*
| `17_early_pipeline_R1.ipynb` | R1.ipynb | early version of the pipeline (exploratory) | *(role inferred from content — please confirm)*
| `18_results_inspector.ipynb` | Untitled41.ipynb | utility: prints the result JSON files on Drive |

## Scripts (`scripts/original/`)

| File | Paper element |
|---|---|
| `leakage_audit.py` | §III-B, §IV-A: DCT/difference hashes over 8 orientations, NCC verification, group-aware split |
| `shape_confound.py` | §III-C, §III-E, §IV-E: Otsu leaf mask, seven geometric descriptors, controls C1–C3 |
| `feature_shape_control.py` | §III-C, §IV-H, Table 7: leaf crop, shape residualisation, shape-matched pairs |
| `localization_eval.py` | §IV-D, Table 3: CorLoc, mIoU, pointing vs baselines |
| `annotate_tool.py` | §IV-D: helper used with localization_eval.py |
| `cross_dataset_confound.py` | §IV-F, Fig. 5: the three controls on five datasets → `artifacts/cross_dataset_confound.json` |
| `overcluster_pipeline.py` | §III-G, §IV-G: clustering with one expert label per cluster, shape control |
| `select_k_and_stability.py` | §IV-G, Table 6: label-free choice of k, seed stability |
| `multiclass_cluster.py` | §IV-G: Hungarian and one-shot naming accuracy, NMI/ARI |
| `few_shot_curve.py` | §IV-J, Table 9: label-efficiency curve and its shape-control column |
| `few_shot_student.py` | earlier student runs (superseded by notebooks 12 and `revision_v1_reproduce.ipynb`) |
| `clutter_robustness.py` | §IV-I: background clutter (first version) |
| `clutter_robustness_fast.py` | §IV-I, Table 8: batched, paired version used for the reported numbers |
| `triage_analysis.py` | original Table 10 (superseded by `analysis/reconcile.py`) |
| `make_all_figures.py` | results figures (notebook 13) |
| `make_results_figures.py` | localisation, triage, cluster and deployment figures (notebook 14) |
| `make_qualitative_figures.py` | attention gallery, head comparison, clutter progression (notebook 14) |

## Not yet in the repository

Imported by the notebooks but not yet supplied. Their results are superseded or are figures, so no
reported number depends on them alone:

| Module | Needed for |
|---|---|
| `few_shot_student_fast.py` | faster variant of few_shot_student.py used by notebooks 10 and 13; superseded by notebook 12 |
| `fig_diagnostics.py` | Figs. 4–5 (notebook 10) |
| `fig1_pipeline.py` | Fig. 2 pipeline diagram (notebook 05) |

## Running the original notebooks

They were written for Colab with the dataset at `/content/drive/MyDrive/Tea Leaf Dataset` and outputs under
`/content/drive/MyDrive/TeaLeaf_DINOv2_Study`. Put the scripts above next to the notebook (or upload them when
the notebook asks) and adjust those two paths. Every number in the revised manuscript can also be regenerated
from the released artifacts without images: see `tools/verify.py` and `notebooks/revision_v1_reproduce.ipynb`.
