# WP1/WP2 baseline benchmark — preserved source-evidence snapshot (2026-10-12)

> **Evidence status: archival baseline, not a fresh experiment.** This section indexes event-level outputs that already exist on GitHub `main` at source revision `72bef39cd71878a30b8af101c60466975ad9fc25`. D2 K–S and B01 Hotelling evidence are from **different datasets and runs**. Do not pool their event counts or describe this document as a newly executed joint EWMA/K–S/Hotelling experiment.

## Reproducible baseline scope

| Case | Subject / stream | Detector and validator | Saved evidence | Status |
| --- | --- | --- | --- | --- |
| D2 | Synthetic D2; subject identifier **N/A** | Stage-I SD-EWMA; Stage-II two-window K–S (TSSD-EWMA) | [Stage-I trace](../metrics/paper2015/d2/stage1_trace.csv), [Stage-I warning event log](../metrics/paper2015/d2/stage2_validations.csv), [Stage-II decision event log](../metrics/paper2015/d2/stage2_validations.csv), [run summary](../metrics/paper2015/d2/summary.json) | Saved event-level K–S decisions; not freshly rerun |
| BCI IV 2B | **B01**, evaluation sessions `04E`, `05E` | Stage-I EWMA on PC1, Stage-II Hotelling T² used by CSE runner | [17 timestamped Stage-I warning rows](../metrics/bci_2b_stage1_warnings.csv), [Table 1 comparison](../metrics/bci_table1_comparison.csv), [older B01 Table 1 result](../metrics/bci_2b_table1_results.csv) | Stage-II **event-level outcomes not archived** in this snapshot; conflicting aggregate CSV counts |

## Effective parameters verified from saved evidence

**D2 (synthetic):** The saved [summary](../metrics/paper2015/d2/summary.json) specifies `estimated_lambda = 0.4` and the Table III **inferred final 1,000-point window**, with reference shift indices `4101, 4201, …, 4901`. The [K–S validation event log](../metrics/paper2015/d2/stage2_validations.csv) records `before_size = 10`, `after_size = 10`, status (confirmed/rejected/skipped/insufficient), alarm and validation times, statistic and p-value where evaluated. The log shows for example alarm `1222`, validation `1232`, `D=0.7`, `p=0.012340600575894691`, `confirmed_shift=True`. The exact runner invocation, random seed, and every runtime knob **are not embedded in the saved summary**; they must not be imputed.

**B01 (real EEG):** The saved [warning rows](../metrics/bci_2b_stage1_warnings.csv) give `subject=B01`, `lambda=0.28`, `control_limit_multiplier=1.96`, `variance_update_mode=always`, `variance_smoothing=0.05`, and `stage_1_alarm=1`, with individual observation index, PC1 value, prediction, EWMA, error, control limits, and session ID. The output has **17 warning rows**: indices `37,48,51,66,76,85,94,116,125,139,146,159,197,219,239,276,300`. The BCI 2B result records `training_trials=280`, `testing_trials=320` for this historical run, with `published_csw=14`, `computed_csw=17`. The 2019 Table 1 [runner](../../scripts/run_bci_table1_reproduction.py) currently defaults to `validation_mode=paper_two_sample`, `validation_window_size=10`, `alpha=0.05`, `minimum_alarm_gap=None`, `pca_components=None`, `components_per_side=1` and seeds `42`; **these are current code defaults, not established historical B01 effective runtime settings**.

### Important B01 discrepancy (must remain visible)

The saved [cross-subject comparison](../metrics/bci_table1_comparison.csv) reports B01 `computed_csv=0`; the separate [B01-only result](../metrics/bci_2b_table1_results.csv) reports `computed_csv=5`. Both give 17 warnings, but they cannot jointly establish the number of confirmed shifts. Neither corresponding timestamped Hotelling test output nor an immutable run manifest is presently linked. Therefore **B01 Stage-II confirmation count is unresolved**. No event rows or p-values have been invented.

## Event-level evidence and interpretation

- **Stage I:** B01 timestamps, values, residuals, estimated variances and dynamically changing thresholds are preserved *in the existing source CSV*; D2 trace and event logs are preserved separately.
- **Stage II K–S:** The D2 per-warning [validation records](../metrics/paper2015/d2/stage2_validations.csv) explicitly distinguish confirmation, rejection, nearby-warning suppression and insufficient pre-window. Missing p-values for skipped/ineligible events are not negative tests.
- **Stage II Hotelling:** B01 per-warning validation status, degrees of freedom, p-value, and validated decision **remain to be exported from an identified rerun**. Do not label all 17 warnings as rejected or confirmed.
- The D2 summary's computed-versus-paper Table III values are **not** a demonstrated numerical reproduction; see [summary](../metrics/paper2015/d2/summary.json).

## Steps required to make a complete runnable frozen experiment

1. Record the exact dataset files and checksums, subject/session partitions, 2A evaluation labels if applicable, code commit, full CLI invocation, dependency versions, and effective per-subject values (PCA dimensions, `lambda`, `L`, variance policy, `alpha`, window sizes, minimum alarm gap).
2. Add a runner export of **all** Stage-I warnings and **all** Stage-II events: `subject,dataset,session_id,alarm_time,validation_time,test,status,n_features,n_before,n_after,df1,df2,statistic,p_value,confirmed_shift`, using nullable fields where a test was not run. Store results in a unique immutable run directory; never overwrite existing experimental outputs.
3. Re-run a fixed B01 configuration once with the archived data and report the new confirmed count **alongside** both historical numbers (0 and 5), not in place of them.
4. Compare against paper results separately from implementation correctness. Emit an automated integrity check ensuring each Stage-II event links to a real Stage-I alarm.

## Current boundary

This output **completes baseline evidence indexing and provenance disclosure**, not the unperformed computation needed to demonstrate a single frozen EWMA/K–S/Hotelling run. Its value is to make the existing experimental records inspectable without silently reconciling incompatible reports. No existing raw data, code, or historical results were modified.
