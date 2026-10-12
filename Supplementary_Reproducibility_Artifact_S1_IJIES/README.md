# Supplementary Reproducibility Artifact S1

**Manuscript:** *Auditing Leakage and Label Conflict in Ultrasound-Based Switchgear Arc-Fault Diagnosis: A Group-Aware CNN-GRU Evaluation*  
**Revision:** Second-round major revision; final IJIES-template 18-page red-font manuscript  
**Purpose:** Machine-readable support for the record-identity audit, corrected grouped evaluation, comparator analysis, uncertainty analysis, sensitivity checks, and timing measurements requested by the reviewers.


## Terminology used in the manuscript
The main article consistently uses **Time Domain** and **Frequency Domain** as the two experimental domains. The phrase **stored one-sided FFT spectrum** is used only to describe the numerical form in which each supplied frequency-domain record is stored; it is not a third domain or a replacement name for the Frequency Domain. The corrected frequency-domain preprocessing therefore maps the supplied one-sided FFT magnitude directly to 128 Mel bands without applying a second STFT.

The source MAT files are **not redistributed**. `source_file_hashes.csv` identifies the exact supplied data/code files used for the audit. This archive is designed to be (i) uploaded to the journal as Supplementary Artifact S1 when an additional-file field is available and (ii) extracted into a public GitHub repository so reviewers can browse the files directly.

## Reproduction
1. Place the six source MAT files (and, for hash verification, the two supplied notebooks) in this folder, **or** set `ARC_AUDIT_DATA_DIR` to the directory containing them.
2. Install the versions listed in `requirements.txt`.
3. Run `run_all.sh` on a CPU environment. The complete run recreates corrected representations, grouped partitions, neural comparators, simple baselines, OOF summaries, sensitivity analyses, and timing files.

The scripts use only training/inner-validation data to fit preprocessing and select checkpoints. Validation and outer-test records are never resampled.

## What can be checked
- `record_identity_audit.csv`: record IDs, source class/index, full SHA-256 identity, cross-label conflict flag, and 0.9999 near-duplicate component.
- `source_file_hashes.csv`: SHA-256 hashes of the supplied source matrices/notebooks used in the audit.
- `initial_pipeline_split_indices.csv`: source-record membership after the original pre-split arcing bootstrap.
- `legacy_holdout_confusion_counts.csv`: time- and frequency-domain legacy test confusion counts underlying the restored main-text legacy trace figures and Table 5.
- `corrected_fixed_split_indices.csv`: deterministic group-safe descriptive split (time selection seed 2260; frequency selection seed 2195).
- `grouped_cv_partitions.csv`: outer/inner grouped partitions, exact group IDs, class labels, and all partition seeds.
- `fold_metrics_main_comparators.csv` / `fold_predictions_main_comparators.csv`: repeated grouped evaluation of CNN-GRU, CNN-LSTM, 1D-CNN, lightweight DSCNN, majority, and a six-feature logistic baseline.
- `pooled_oof_predictions.csv` / `pooled_oof_summary.csv`: record-level out-of-fold probabilities averaged over two repetitions and pooled confusion information.
- `identity_cluster_bootstrap_uncertainty.csv`: 10,000-replicate uncertainty intervals from resampling exact identity groups rather than treating ten fold scores as independent.
- `paired_outer_fold_differences.csv`: paired comparator-minus-CNN-GRU differences on the same outer folds.
- `initialization_sensitivity_*.csv`: three initialization seeds on one fixed outer split, separated from split variability.
- `conflict_policy_supports.csv` / `frequency_conflict_exclusion_cnn_gru.csv`: predeclared cross-label-conflict sensitivity.
- `near_duplicate_threshold_sensitivity.csv` / `near_duplicate_grouping_frequency_cnn_gru.csv`: threshold and expanded-group sensitivity.
- `matched_controlled_fast.csv` / `matched_controlled_fast_summary.csv`: same-record representation/recurrent-cell sensitivity outputs.
- `config.json`: initial and corrected representations, normalization, split rules, training rules, seeds, bootstrap, and timing protocol.
- `timing_measurements.csv` and `timing_protocol.md`: CPU timing measurements and exact timing protocol.
- `prepare_representations.py`, `train_models.py`, `postprocess_results.py`, and sensitivity scripts: executable reproduction code.

## Mapping to the final 18-page manuscript
- **Sections 3.2-3.4 / Table 1:** source composition, initial and corrected splits, exact identity grouping.
- **Section 3.3 / Table 2:** corrected representation definitions and notation.
- **Section 3.5 / Figs. 4-5 / Table 3:** corrected CNN-GRU architectures and parameterization.
- **Section 3.6 / Table 4:** initial-versus-corrected training and evaluation settings.
- **Section 4.1 / Figs. 6-10 / Table 5:** legacy notebook training/confusion traces plus the legacy-versus-corrected audit contrast.
- **Section 4.2 / Tables 6-7:** data-integrity counts and repeated comparator evaluation.
- **Section 4.3 / Figs. 11-12 / Table 8:** pooled OOF confusion information and identity-cluster bootstrap uncertainty.
- **Section 4.4 / Table 9 / Figs. 13-14:** conflict, initialization, near-duplicate, and traceability sensitivities.
- **Section 5 / Tables 10-11:** contextual interpretation and evidence hierarchy.
- **Code and Reproducibility Artifact:** release scope, data-redistribution boundary, and GitHub/S1 instructions.

## Conflict handling
The primary analysis does not relabel contradictory exact groups. It retains their recorded labels while constraining every exact identity group to a single fold. The exclusion sensitivity removes all exact groups containing both labels before fitting; this policy is defined solely from identity and labels, not from test performance. In the time domain that exclusion leaves only 3 arcing records in 2 exact groups, so a stable five-fold positive-class evaluation is not claimed.

## Near-duplicate limitation
The 0.999/0.9999/0.99999 absolute-correlation checks are a documented sensitivity analysis, not a reconstruction of unavailable acquisition sessions. Raw acquisition IDs and original audio/session metadata were not supplied.

## Dataset permissions
The processed MAT files, raw audio, and source notebooks are not included in the public artifact because redistribution permission was not established. The artifact is machine-checkable for released metadata/results and executable when the permitted source matrices are supplied by the dataset holder/authors.

## GitHub release
Upload the **extracted files** to the repository root rather than publishing only the ZIP. This keeps the README, scripts, configurations, hashes, split indices, fold predictions, sensitivities, and timing protocol browser-accessible. The exact journal ZIP can additionally be attached to a tagged GitHub Release as the immutable S1 snapshot.

See `GITHUB_UPLOAD_GUIDE.md` and `ARTIFACT_MANIFEST.md` for the final upload checklist.
