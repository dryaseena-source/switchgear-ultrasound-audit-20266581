# Artifact Manifest

This manifest groups the public/reviewable files in Supplementary Reproducibility Artifact S1. Source MAT files and raw audio are intentionally excluded.

## Core documentation
- `README.md`
- `GITHUB_UPLOAD_GUIDE.md`
- `CITATION.cff`
- `config.json`
- `requirements.txt`
- `.gitignore`
- `run_all.sh`
- `MANUSCRIPT_MAPPING_EXPANDED.md`

## Provenance and partitions
- `source_file_hashes.csv`
- `record_identity_audit.csv`
- `initial_pipeline_split_indices.csv`
- `legacy_holdout_confusion_counts.csv`
- `corrected_fixed_split_indices.csv`
- `grouped_cv_partitions.csv`

## Primary repeated comparison
- `fold_metrics_main_comparators.csv`
- `fold_predictions_main_comparators.csv`
- `repeated_comparator_summary.csv`
- `paired_outer_fold_differences.csv`

## Pooled OOF and uncertainty
- `pooled_oof_predictions.csv`
- `pooled_oof_summary.csv`
- `identity_cluster_bootstrap_uncertainty.csv`
- `group_multiplicity_sensitivity.csv`

## Sensitivity analyses
- `conflict_policy_supports.csv`
- `frequency_conflict_exclusion_cnn_gru.csv`
- `near_duplicate_threshold_sensitivity.csv`
- `near_duplicate_grouping_frequency_cnn_gru.csv`
- `initialization_sensitivity_time.csv`
- `initialization_sensitivity_frequency.csv`
- `initialization_sensitivity_summary.csv`
- `matched_controlled_fast.csv`
- `matched_controlled_fast_summary.csv`

## Timing
- `timing_measurements.csv`
- `timing_protocol.md`
- `timing.py`

## Executable code
- `prepare_representations.py`
- `train_models.py`
- `postprocess_results.py`
- `rev2_train_models.py`
- `rev2_postprocess.py`
- `rev2_conflict_freq.py`
- `rev2_near_freq.py`
- `rev2_init_sensitivity.py`

## Additional fold-level outputs
- `fold_metrics_time_CNN-GRU.csv`
- `fold_predictions_time_CNN-GRU.csv`
- `fold_metrics_time_CNN-LSTM.csv`
- `fold_predictions_time_CNN-LSTM.csv`
- `fold_metrics_time_1D-CNN.csv`
- `fold_predictions_time_1D-CNN.csv`
- `fold_metrics_time_GRU.csv`
- `fold_predictions_time_GRU.csv`
- `fold_metrics_time_Light-DSCNN.csv`
- `fold_predictions_time_Light-DSCNN.csv`
- `fold_metrics_time_simple.csv`
- `fold_predictions_time_simple.csv`
- `fold_metrics_frequency_CNN-GRU.csv`
- `fold_predictions_frequency_CNN-GRU.csv`
- `fold_metrics_frequency_CNN-LSTM.csv`
- `fold_predictions_frequency_CNN-LSTM.csv`
- `fold_metrics_frequency_1D-CNN.csv`
- `fold_predictions_frequency_1D-CNN.csv`
- `fold_metrics_frequency_Light-DSCNN.csv`
- `fold_predictions_frequency_Light-DSCNN.csv`
- `fold_metrics_frequency_simple.csv`
- `fold_predictions_frequency_simple.csv`
- `controlled_baseline_summary.csv`
