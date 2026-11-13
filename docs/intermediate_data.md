# Intermediate data

The computation commands create reusable result bundles under
`program_v04/results/`. Each bundle contains Parquet panels and a JSON
completion record with input hashes, settings, source identity, and output
hashes. Individual table and figure commands read these bundles and do not
fit models.

| Bundle | Contents | Generating command |
| --- | --- | --- |
| `results/main/section2/` | Monthly binary signals and summary statistics for Figure 1 and Table 1 | `python -m program_v04.main.compute_main --input-mode public` |
| `results/shared/revised_main/` | Main forecasts, scores, and feature importances | Same main computation command |
| `results/shared/revised_extended/` | Extended-model forecasts, scores, and feature importances, including Random Forest | Same main computation command |
| `results/main/revised_probit_spread/` | Term-spread Probit forecasts and scores | Same main computation command |
| `results/main/revised_parameter_free/` | Share-of-binary-states forecasts and scores | Same main computation command |
| Appendix revised-family bundles | Threshold, smoothing, no-lag, parsimonious, alternative-subset, and expanding-threshold results | `python -m program_v04.appendix.compute_appendix --input-mode public` |
| Appendix real-time bundles | Paired revised-data and vintage forecasts and scores | Same appendix computation command |
| `results/appendix/derived/` | Sampling uncertainty, pre-2020 comparisons, and cumulative Brier differences | Same appendix computation command |
| `results/appendix/decision_wealth/` | CRRA investment summaries and monthly accounts | Same appendix command, or `python -m program_v04.appendix.compute_decision_wealth` after computing the main forecasts |
| `results/appendix/simulation/` | Simulated forecasts, replication scores, and summaries | Same appendix command, or `python -m program_v04.appendix.compute_appendix --simulation-only --input-mode public` |

Paths in this table are relative to `program_v04/`. The exact dependency paths
for every artifact are listed in [ARTIFACTS.md](../program_v04/docs/ARTIFACTS.md).
Preprocessing matrices are constructed in memory from the declared raw inputs;
they are not a separate saved dataset required by the artifact commands.

The public data-free edition omits empirical intermediate panels. Its full
computation creates the 35 numerical panels and 16 metadata files before
artifact generation. There is no saved-results shortcut on a fresh extraction.
After the full run, retain these outputs to regenerate tables and figures
without refitting. Their omission and the provider data rights are explained
in [data_rights.md](data_rights.md).

The separately retained confidential journal edition includes these files
under `frozen/results/`; its default launcher copies them into
`program_v04/results/` for artifact generation and checks they remain unchanged.
That edition and its saved-results shortcut are not part of the data-free
final submission packet.

The small CSV, LaTeX, and PDF files under `reference/output/` record the paper's
reported results. They are included for comparison and cannot replace the
forecast and score panels used by the generators.
