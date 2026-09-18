# Shuaa measured training report

Source commit: `42e0b4610206e80a5fc2af62efc95a5adbfc5657`. Split hash: `043313273c135d80e4a298717c19253145ec1156eb99dc69441b78dcbd4139f5`.

Campaign status: **complete**. These are validation results; test was not evaluated. Time-limited runs may have stopped before convergence.

| Run | Epochs | Stop reason | Validation accuracy | Validation Macro-F1 |
|---|---:|---|---:|---:|
| efficientnetv2_ce | 35 | completed | 0.838720 | 0.710192 |
| convnextv2_ce | 19 | early_stopped | 0.845052 | 0.726310 |
| efficientnetv2_weighted | 35 | completed | 0.827058 | 0.703563 |
| convnextv2_weighted | 35 | completed | 0.844718 | 0.737417 |
| convnextv2_ce_seed43 | 35 | completed | 0.848384 | 0.724903 |

Selected: `probability_mean` using convnextv2_ce_seed43, convnextv2_ce.

Full methodology and limitations: `research/methodology.md`. All per-epoch logs and resumable checkpoints are in `runs/`. ONNX parity report: `model.parity.json`.
