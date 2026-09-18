# Shuaa: next experiments after the first measured campaign

Research date: 2026-09-18. Recommendations below are hypotheses, not measured improvements.

## Measured baseline and error analysis

The selected equal-probability ensemble of ConvNeXt V2 Tiny seeds 42 and 43 achieved validation accuracy **85.0716%**, Macro-F1 **73.7819%**, and 448 errors on 3,001 images. Best single-model accuracy was 84.8384%. Test remains untouched. Evidence: [provenance](runs/training-20260918-024537/provenance.json) and [campaign](runs/training-20260918-024537/campaign.json).

Validation recall identifies concrete weaknesses: Soiling 9/31 (29.0%), Cell-Multi 100/193 (51.8%), Offline-Module 84/124 (67.7%), Vegetation 174/246 (70.7%). No-Anomaly is 1,470/1,500 (98.0%); aggregate accuracy alone conceals minority-class weaknesses. The four named weak classes account for 227 of 448 errors. Inspect training examples and validation error categories without relabeling validation to improve a score. Any independently justified label correction needs a versioned benchmark and reevaluation of all entries.

## Prioritized experiments

1. **ConvNeXt V2 Tiny with ImageNet-22K fine-tuning weights.** Compare `convnextv2_tiny.fcmae_ft_in22k_in1k` against the current `convnextv2_tiny.fcmae_ft_in1k` at the same 224 resolution, split, seed and optimizer settings. This isolates weight initialization without increasing model size. The [official model family](https://github.com/facebookresearch/ConvNeXt-V2) supplies 22K variants; the [timm model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k) confirms the exact checkpoint. Its source benchmark gains are not a prediction for thermal images.
2. **Tune the current model before scaling up.** Compare learning rates 3e-5 and 1e-4 with the same fixed budget and seeds; test label smoothing 0.05 separately against zero. Then a separate modest mixup trial (alpha 0.1). [Label smoothing paper](https://arxiv.org/abs/1906.02629) and [mixup paper](https://arxiv.org/abs/1710.09412) motivate regularization, but combining all changes at once would hide their effects. Mixing thermal patterns could hurt defect discrimination, so keep only measured gains.
3. **Address imbalance gently.** Naive weighted CE already reduced accuracy (ConvNeXt seed42: 84.47% versus 84.51% CE; EfficientNet: 82.71% versus 83.87%). Compare effective-number class weighting against CE, using counts from training only, as proposed in [Class-Balanced Loss](https://arxiv.org/abs/1901.05555). Record both accuracy and per-class recall; do not assume weighting improves the ranking metric.
4. **Try a genuinely different representation: DINOv2 ViT-S/14.** Begin with a frozen encoder and trained 12-class head, then unfreeze the final blocks only if validation supports it. The [paper](https://arxiv.org/abs/2304.07193) and [official implementation](https://github.com/facebookresearch/dinov2) support transferable visual features. Natural-image pretraining does not guarantee gains on tiny thermal crops. Verify input normalization and ONNX parity before submission.
5. **Optional capacity and inference experiments.** ConvNeXt V2 Base with 22K weights is an available larger candidate if GPU budget remains. Separately compare identity versus mean probabilities over the four horizontal/vertical flip combinations, since flips are already training augmentations. This test-time augmentation is our proposed ablation, not an established result on this dataset; it multiplies inference cost by four. Embed accepted transforms inside ONNX to preserve the shared evaluator contract.

## Experimental controls and decision rule

Keep the existing duplicate-grouped split and excluded conflicting labels fixed. Reserve test until one final candidate is locked. Run one-change comparisons first, repeat promising configurations with a second seed, and compare both single-model and ensemble scores. Record all trials, training time and inference cost. Rank by validation accuracy, then Macro-F1; tiny gains on the repeatedly used validation set require caution. Use paired validation predictions to inspect whether a candidate fixes the baseline's errors before adding it to an ensemble.

The original images are only 24 by 40 pixels. Upscaling to 384 cannot restore missing information and should not be the first use of GPU budget. Avoid heavy crops or contrast changes that might remove faults or alter thermal relationships. Any preprocessing experiment must be applied identically at training/export/evaluation time.

Published PV classification scores are not directly comparable unless class count, split, duplicate handling and metric definitions match ours. No claimed target accuracy or global-best model is justified by this research. No second campaign has been launched as part of this research note.
