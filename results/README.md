# Raw results

| Folder | Source | Used in the paper |
| --- | --- | --- |
| `local/strengthening_results.json` | `experiments/run_strengthening_experiments.py` | Sections 6.6, 6.11 and 6.12, encrypted silence |
| `local/bench_aead_baselines.json` | `experiments/bench_aead_baselines.py`, five interleaved runs with AES instructions available (`pair_hw_*`) and disabled (`pair_noaes_*`); `run*` are earlier sessions | Table 10 throughput |
| `nist_sts/` | official NIST STS 2.1.2 on 1,000 sequences of encrypted audio and of encrypted silence (`finalAnalysisReport_*.txt`, summarised in `summary_*.json`) | Table 6 |
| `local/results.json` | `experiments/run_local_experiments.py`, one 2.1 GHz Xeon core | Tables 3-6, ablation (Table 5), nonce reuse (6.6), tamper detection (6.7), throughput (Figure 5) |
| `kaggle_run1/` | first Kaggle run (T4 GPU) | **Only** `partD_nist.json` (Table 7) and `partC_throughput.json` (Section 6.8). The standard-dataset and attack results of this run are not used: the three datasets were read as one, and ESC-50 duplicates crossed the train/test split. The paper reports this in Section 6.10. |
| `kaggle_run2/` | Kaggle run with per-dataset evaluation, duplicate removal and source-disjoint splits | Table 8 (`partA_*`), Table 9 and Figure 6 (`partB_dl_attack.json`, recomputed against pure guessing in `partB_uniform_baseline.json`), throughput range (`partC_throughput.json`) |

`kaggle_run2/environment.json` lists the one unreadable dataset file that was skipped (GTZAN `jazz.00054.wav`).
