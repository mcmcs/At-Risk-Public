# Artifact Contracts

This file is generated from the stable artifact registry and the current
manuscript float order. Do not edit it by hand. Each artifact command reads
stored, fingerprint-validated dependencies and never estimates a model.

The closed inventory contains 26 numbered artifacts: 10 main-profile and 16 appendix-profile.

| Manuscript object | Profile | Class | Stable ID | Label | Module |
|---|---|---|---|---|---|
| Figure 1 | main | computational | `fig_single_binary_examples` | `fig:single_binary_examples` | `program_v04.main.artifacts.generate_figure01` |
| Table 1 | main | computational | `tab_single_binary_summary` | `tab:single_binary_summary` | `program_v04.main.artifacts.generate_table01` |
| Table 2 | main | computational | `tab_zt_vs_xt` | `tab:zt_vs_xt` | `program_v04.main.artifacts.generate_table02` |
| Table 3 | main | computational | `tab_aggregation_comparison` | `tab:aggregation_comparison` | `program_v04.main.artifacts.generate_table03` |
| Table 4 | main | computational | `tab_encompassing_tests` | `tab:encompassing_tests` | `program_v04.main.artifacts.generate_table04` |
| Figure 2 | main | computational | `fig_main_probability_plots` | `fig:main_probability_plots` | `program_v04.main.artifacts.generate_figure02` |
| Table 5 | main | computational | `tab_sse_decomposition_h3` | `tab:sse_decomposition_h3` | `program_v04.main.artifacts.generate_table05` |
| Figure 3 | main | computational | `fig_disagreement_h3` | `fig:disagreement_h3` | `program_v04.main.artifacts.generate_figure03` |
| Table 6 | main | computational | `tab_feature_importance` | `tab:feature_importance` | `program_v04.main.artifacts.generate_table06` |
| Table 7 | main | computational | `tab_sector_contribution` | `tab:sector_contribution` | `program_v04.main.artifacts.generate_table07` |
| Table 8 | appendix | computational | `tab_zt_vs_xt_roc` | `tab:zt_vs_xt_roc` | `program_v04.appendix.artifacts.generate_table08` |
| Table 9 | appendix | computational | `tab_robustness_thresholds` | `tab:robustness_thresholds` | `program_v04.appendix.artifacts.generate_table09` |
| Table 10 | appendix | computational | `tab_robustness_ma` | `tab:robustness_ma` | `program_v04.appendix.artifacts.generate_table10` |
| Table 11 | appendix | computational | `tab_extended_models` | `tab:extended_models` | `program_v04.appendix.artifacts.generate_table11` |
| Table 12 | appendix | computational | `tab_zt_vs_xt_unlagged` | `tab:zt_vs_xt_unlagged` | `program_v04.appendix.artifacts.generate_table12` |
| Table 13 | appendix | computational | `tab_subset` | `tab:subset` | `program_v04.appendix.artifacts.generate_table13` |
| Table 14 | appendix | computational | `tab_subset_extra` | `tab:subset_extra` | `program_v04.appendix.artifacts.generate_table14` |
| Table 15 | appendix | computational | `tab_realtime` | `tab:realtime` | `program_v04.appendix.artifacts.generate_table15` |
| Table 16 | appendix | computational | `tab_sampling_uncertainty` | `tab:sampling_uncertainty` | `program_v04.appendix.artifacts.generate_table16` |
| Table 17 | appendix | computational | `tab_pre2020_robustness` | `tab:pre2020_robustness` | `program_v04.appendix.artifacts.generate_table17` |
| Figure 4 | appendix | computational | `fig_cumulative_brier` | `fig:cumulative_brier` | `program_v04.appendix.artifacts.generate_figure04` |
| Table 18 | appendix | computational | `tab_asymmetric_loss` | `tab:asymmetric_loss` | `program_v04.appendix.artifacts.generate_table18` |
| Table 19 | appendix | computational | `tab_expanding_threshold` | `tab:expanding_threshold` | `program_v04.appendix.artifacts.generate_table19` |
| Table 20 | appendix | simulation | `tab_mc_tailrisk` | `tab:mc_tailrisk` | `program_v04.appendix.artifacts.generate_table20` |
| Table 21 | appendix | documentary | `tab_lit_survey` | `tab:lit_survey` | `program_v04.appendix.artifacts.generate_table21` |
| Table 22 | appendix | documentary | `tab_lit_survey_nonijf` | `tab:lit_survey_nonijf` | `program_v04.appendix.artifacts.generate_table22` |

## Per-artifact dependencies and outputs

Run a module with `--help` to see this same contract at the command line.

### Figure 1 — `fig:single_binary_examples`

- Stable ID: `fig_single_binary_examples`
- Profile/class: `main` / `computational`
- Manuscript section: Illustrative Example
- Command: `python -m program_v04.main.artifacts.generate_figure01`
- Required dependencies:
  - `results/main/section2/signal_panel.parquet`
  - `results/main/section2/summary.parquet`
  - `results/main/section2/metadata.json`
- Outputs:
  - `output/main/figures/section2_binary_signals.pdf`

### Table 1 — `tab:single_binary_summary`

- Stable ID: `tab_single_binary_summary`
- Profile/class: `main` / `computational`
- Manuscript section: Illustrative Example
- Command: `python -m program_v04.main.artifacts.generate_table01`
- Required dependencies:
  - `results/main/section2/summary.parquet`
  - `results/main/section2/metadata.json`
- Outputs:
  - `output/main/tables/table01_single_binary_summary.csv`
  - `output/main/tables/table01_single_binary_summary.tex`

### Table 2 — `tab:zt_vs_xt`

- Stable ID: `tab_zt_vs_xt`
- Profile/class: `main` / `computational`
- Manuscript section: Main Empirical Findings
- Command: `python -m program_v04.main.artifacts.generate_table02`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/shared/revised_main/forecasts.parquet`
  - `results/shared/revised_extended/metrics.parquet`
  - `results/shared/revised_extended/forecasts.parquet`
  - `results/main/revised_probit_spread/metrics.parquet`
- Outputs:
  - `output/main/tables/table02_zt_vs_benchmarks.csv`
  - `output/main/tables/table02_zt_vs_benchmarks.tex`

### Table 3 — `tab:aggregation_comparison`

- Stable ID: `tab_aggregation_comparison`
- Profile/class: `main` / `computational`
- Manuscript section: Main Empirical Findings
- Command: `python -m program_v04.main.artifacts.generate_table03`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/main/revised_parameter_free/metrics.parquet`
- Outputs:
  - `output/main/tables/table03_aggregation_comparison.csv`
  - `output/main/tables/table03_aggregation_comparison.tex`

### Table 4 — `tab:encompassing_tests`

- Stable ID: `tab_encompassing_tests`
- Profile/class: `main` / `computational`
- Manuscript section: Main Empirical Findings
- Command: `python -m program_v04.main.artifacts.generate_table04`
- Required dependencies:
  - `results/shared/revised_main/forecasts.parquet`
  - `results/shared/revised_extended/forecasts.parquet`
- Outputs:
  - `output/main/tables/table04_encompassing_tests.csv`
  - `output/main/tables/table04_encompassing_tests.tex`

### Figure 2 — `fig:main_probability_plots`

- Stable ID: `fig_main_probability_plots`
- Profile/class: `main` / `computational`
- Manuscript section: Understanding Drivers
- Command: `python -m program_v04.main.artifacts.generate_figure02`
- Required dependencies:
  - `results/shared/revised_main/forecasts.parquet`
- Outputs:
  - `output/main/figures/Zt_vs_Xt_h3.pdf`
  - `output/main/figures/Zt_vs_Xt_h6.pdf`
  - `output/main/figures/Zt_vs_Xt_h12.pdf`

### Table 5 — `tab:sse_decomposition_h3`

- Stable ID: `tab_sse_decomposition_h3`
- Profile/class: `main` / `computational`
- Manuscript section: Understanding Drivers
- Command: `python -m program_v04.main.artifacts.generate_table05`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
- Outputs:
  - `output/main/tables/table05_mse_decomposition_h3.csv`
  - `output/main/tables/table05_mse_decomposition_h3.tex`

### Figure 3 — `fig:disagreement_h3`

- Stable ID: `fig_disagreement_h3`
- Profile/class: `main` / `computational`
- Manuscript section: Understanding Drivers
- Command: `python -m program_v04.main.artifacts.generate_figure03`
- Required dependencies:
  - `results/shared/revised_main/forecasts.parquet`
- Outputs:
  - `output/main/figures/forecast_disagreement_h3.pdf`

### Table 6 — `tab:feature_importance`

- Stable ID: `tab_feature_importance`
- Profile/class: `main` / `computational`
- Manuscript section: Understanding Drivers
- Command: `python -m program_v04.main.artifacts.generate_table06`
- Required dependencies:
  - `results/shared/revised_main/importances.parquet`
- Outputs:
  - `output/main/tables/table06_feature_importance.csv`
  - `output/main/tables/table06_feature_importance.tex`

### Table 7 — `tab:sector_contribution`

- Stable ID: `tab_sector_contribution`
- Profile/class: `main` / `computational`
- Manuscript section: Understanding Drivers
- Command: `python -m program_v04.main.artifacts.generate_table07`
- Required dependencies:
  - `results/shared/revised_main/importances.parquet`
- Outputs:
  - `output/main/tables/table07_sector_contribution.csv`
  - `output/main/tables/table07_sector_contribution.tex`

### Table 8 — `tab:zt_vs_xt_roc`

- Stable ID: `tab_zt_vs_xt_roc`
- Profile/class: `appendix` / `computational`
- Manuscript section: Additional Metrics
- Command: `python -m program_v04.appendix.artifacts.generate_table08`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table08_zt_vs_xt_roc.csv`
  - `output/appendix/tables/table08_zt_vs_xt_roc.tex`

### Table 9 — `tab:robustness_thresholds`

- Stable ID: `tab_robustness_thresholds`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table09`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/appendix/revised_threshold/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table09_robustness_thresholds.csv`
  - `output/appendix/tables/table09_robustness_thresholds.tex`

### Table 10 — `tab:robustness_ma`

- Stable ID: `tab_robustness_ma`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table10`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/appendix/revised_moving_average/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table10_robustness_ma.csv`
  - `output/appendix/tables/table10_robustness_ma.tex`

### Table 11 — `tab:extended_models`

- Stable ID: `tab_extended_models`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table11`
- Required dependencies:
  - `results/shared/revised_extended/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table11_extended_models.csv`
  - `output/appendix/tables/table11_extended_models.tex`

### Table 12 — `tab:zt_vs_xt_unlagged`

- Stable ID: `tab_zt_vs_xt_unlagged`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table12`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/appendix/revised_nolag/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table12_zt_vs_xt_unlagged.csv`
  - `output/appendix/tables/table12_zt_vs_xt_unlagged.tex`

### Table 13 — `tab:subset`

- Stable ID: `tab_subset`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table13`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/appendix/revised_parsimonious/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table13_subset.csv`
  - `output/appendix/tables/table13_subset.tex`

### Table 14 — `tab:subset_extra`

- Stable ID: `tab_subset_extra`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table14`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/appendix/revised_alternative_subsets/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table14_subset_extra.csv`
  - `output/appendix/tables/table14_subset_extra.tex`

### Table 15 — `tab:realtime`

- Stable ID: `tab_realtime`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table15`
- Required dependencies:
  - `results/appendix/realtime_main/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table15_realtime.csv`
  - `output/appendix/tables/table15_realtime.tex`

### Table 16 — `tab:sampling_uncertainty`

- Stable ID: `tab_sampling_uncertainty`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table16`
- Required dependencies:
  - `results/appendix/derived/bootstrap.parquet`
- Outputs:
  - `output/appendix/tables/table16_sampling_uncertainty.csv`
  - `output/appendix/tables/table16_sampling_uncertainty.tex`

### Table 17 — `tab:pre2020_robustness`

- Stable ID: `tab_pre2020_robustness`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table17`
- Required dependencies:
  - `results/appendix/derived/pre2020.parquet`
- Outputs:
  - `output/appendix/tables/table17_pre2020_robustness.csv`
  - `output/appendix/tables/table17_pre2020_robustness.tex`

### Figure 4 — `fig:cumulative_brier`

- Stable ID: `fig_cumulative_brier`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_figure04`
- Required dependencies:
  - `results/appendix/derived/cumulative_brier.parquet`
- Outputs:
  - `output/appendix/figures/cumulative_brier_differentials.pdf`

### Table 18 — `tab:asymmetric_loss`

- Stable ID: `tab_asymmetric_loss`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table18`
- Required dependencies:
  - `results/appendix/decision_wealth/summary.parquet`
  - `results/appendix/decision_wealth/wealth_paths.parquet`
- Outputs:
  - `output/appendix/tables/table18_asymmetric_loss.csv`
  - `output/appendix/tables/table18_asymmetric_loss.tex`

### Table 19 — `tab:expanding_threshold`

- Stable ID: `tab_expanding_threshold`
- Profile/class: `appendix` / `computational`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table19`
- Required dependencies:
  - `results/shared/revised_main/metrics.parquet`
  - `results/shared/revised_main/metadata.json`
  - `results/appendix/revised_expanding_threshold/metrics.parquet`
  - `results/appendix/revised_expanding_threshold/metadata.json`
- Outputs:
  - `output/appendix/tables/table19_expanding_threshold.csv`
  - `output/appendix/tables/table19_expanding_threshold.tex`

### Table 20 — `tab:mc_tailrisk`

- Stable ID: `tab_mc_tailrisk`
- Profile/class: `appendix` / `simulation`
- Manuscript section: Robustness checks
- Command: `python -m program_v04.appendix.artifacts.generate_table20`
- Required dependencies:
  - `results/appendix/simulation/metrics.parquet`
- Outputs:
  - `output/appendix/tables/table20_mc_tailrisk.csv`
  - `output/appendix/tables/table20_mc_tailrisk.tex`

### Table 21 — `tab:lit_survey`

- Stable ID: `tab_lit_survey`
- Profile/class: `appendix` / `documentary`
- Manuscript section: Predictor survey
- Command: `python -m program_v04.appendix.artifacts.generate_table21`
- Required dependencies:
  - `data/documentary/literature_survey_ijf.csv`
- Outputs:
  - `output/appendix/tables/table21_lit_survey.csv`
  - `output/appendix/tables/table21_lit_survey.tex`

### Table 22 — `tab:lit_survey_nonijf`

- Stable ID: `tab_lit_survey_nonijf`
- Profile/class: `appendix` / `documentary`
- Manuscript section: Predictor survey
- Command: `python -m program_v04.appendix.artifacts.generate_table22`
- Required dependencies:
  - `data/documentary/literature_survey_nonijf.csv`
- Outputs:
  - `output/appendix/tables/table22_lit_survey_nonijf.csv`
  - `output/appendix/tables/table22_lit_survey_nonijf.tex`
