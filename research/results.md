# Shuaa results status

The first Kaggle campaign completed on 2026-09-18. Its selected two-seed ConvNeXt V2 Tiny ensemble achieved 85.0716% validation accuracy and 73.7819% Macro-F1 in training. The [measured report](runs/training-20260918-024537/measured-report.md) and [provenance](runs/training-20260918-024537/provenance.json) preserve the evidence. The submitted ONNX is evaluated independently for the live leaderboard; test remains untouched. See the [next-experiment research](improvement-plan.md) for proposed improvements, which have not yet been trained.

No trained accuracy is hard-coded in this repository. Read the live benchmark records for independently evaluated scores.

The training runner generates the measured report, epoch logs, validation predictions, ONNX parity check and provenance in `shuaa-output/` after an actual Kaggle run. Until those artifacts exist, the corresponding experiment is pending. A CPU synthetic-fixture smoke test verifies the implementation only and is not a solar-dataset result.

Report each real run with its architecture, seed, weighted/unweighted loss, epochs, stopping reason, validation accuracy and Macro-F1. Keep final test scores separate and record the locked finalist IDs and split hash. Include limitations from [the methodology](methodology.md) when presenting results to the instructor.
