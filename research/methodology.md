# Shuaa: paper-backed solar module anomaly classification

## Research question and evidence

Which of two pretrained convolutional architectures performs best on the team's fixed InfraredSolarModules validation split, under the same GPU budget, preprocessing and evaluation protocol? The deployed model runs on a server or laptop; there is no drone memory constraint. The selected architectures are practical candidates, not a claim of globally optimal accuracy.

1. Tan and Le, **EfficientNetV2: Smaller Models and Faster Training**, ICML 2021, PMLR 139:10096–10106. [Primary paper](https://proceedings.mlr.press/v139/tan21a.html). The paper introduces training-aware architecture search and Fused-MBConv. Shuaa transfers the small ImageNet-1k model, `tf_efficientnetv2_s.in1k`.
2. Woo et al., **ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders**, 2023. [Primary paper](https://arxiv.org/abs/2301.00808). Its fully convolutional masked-autoencoder pretraining and Global Response Normalization motivate the second candidate, `convnextv2_tiny.fcmae_ft_in1k`.
3. Raptor Maps, **InfraredSolarModules**, originally published at the ICLR 2020 AI for Earth Sciences workshop. [Dataset and source paper link](https://github.com/RaptorMaps/InfraredSolarModules). There are 20,000 infrared module crops, 24 by 40 pixels, covering 11 anomaly classes plus No-Anomaly. The 10,000 nominal images make overall accuracy sensitive to class imbalance.

This is transfer learning from the cited models, not an exact reproduction of either paper's ImageNet training recipe. In particular, training uses a fixed 224-pixel input and does not reproduce EfficientNetV2 progressive-resolution training. ImageNet RGB pretraining is a starting point for thermal imagery; its usefulness must be measured on this dataset.

## Fixed benchmark and leakage controls

The dataset archive is downloaded from a pinned Git commit and verified with SHA-256. The committed `benchmark/split.json` specifies the exact sample identities, class order, split hash, seed, archive identity and duplicate audit. Six identical-pixel pairs have contradictory labels. All 12 affected images are quarantined before any split or model fitting, leaving 19,988 eligible images. Their IDs and original labels remain in the audit. The remaining identical-pixel groups stay together. The fixed partitions contain 13,991 training, 3,001 validation, and 2,996 test images (approximately 70%/15%/15% per class). Training verifies the manifest hash and persists both its semantic hash and its exact file checksum in every checkpoint. Near-duplicate detection is not implemented; only exact decoded-pixel duplicates are grouped.

Only train and validation partitions can be loaded by the tuning dataset class. Model selection, stopping and ensemble decisions use validation only. Test is reserved for the independently locked final evaluation after one candidate per participant has been fixed. All source labels are public, so this is a methodological holdout, not a secret test.

Exact-pixel grouping does not detect all near duplicates or guarantee separation by solar farm, inspection session or physical module. The source metadata does not provide those grouping identifiers. These limitations must accompany accuracy claims; a deployment claim requires an external farm/inspection holdout.

## Training protocol

- Convert to RGB (grayscale intensities are repeated across channels), resize the complete image to 224×224 with PIL bilinear interpolation, divide by 255, and normalize by ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. The resize preserves all pixels but changes aspect ratio and does not add thermal detail.
- Apply independent horizontal and vertical flips with probability 0.5 on training images only. No crop, color jitter, test-time augmentation or validation augmentation is used. Flips assume the label is invariant to image orientation.
- Freeze the backbone and batch-normalization statistics for two epochs and train the classification head; then unfreeze all parameters. Use AdamW, learning rate 0.0001, weight decay 0.01, cosine decay, gradient clipping at 5, mixed precision on CUDA, batch size 32 and two-step gradient accumulation.
- Train up to 35 epochs, stop after eight epochs without improvement, and save `last.pt` after each epoch. Save `best.pt` and validation probabilities when accuracy improves, or Macro-F1 improves at the same accuracy. Checkpoints contain optimizer, scheduler, scaler, RNG states, data-loader generator, full history and split identity; resume occurs at an epoch boundary.
- Compare ordinary cross-entropy and inverse-frequency weighted cross-entropy for each architecture. Weights are `N/(12*n_class)`, calculated from training labels only; no oversampling is used.
- Start all four conditions with seed 42; repeat the best initial configuration with seed 43. Two seeds provide a limited stability check, not a reliable confidence interval.
- Compare the arithmetic mean probabilities of the two best completed models against the best single model, using the same validation metrics. Export an ensemble only when its validation ranking strictly improves. This extra selection uses validation and must be disclosed when reporting generalization.

The initial campaign caps each configuration at two GPU hours and the entire session at 11 hours, reserving room within the Kaggle session for setup and export. Time limits are checked at epoch boundaries, so the current epoch is completed. The campaign stops launching runs when its remaining budget is insufficient. A budget-limited model may not have converged. Free GPU availability and notebook limits are external constraints; two calendar days do not imply 48 GPU hours.

## Measurement and artifacts

Accuracy is the primary ranking metric. Macro-F1 is the tie-breaker; per-class recall and a 12×12 confusion matrix reveal minority-class failures. Both test and validation results must be clearly labeled and must not be pooled into one ranking. Repeated validation tuning creates selection bias; report the separate final test result after locking candidates.

The notebook saves actual per-epoch histories, selected-model validation probabilities, seeds, configuration, source commit, dataset identity, software versions, GPU type and run timing. The exported graph includes softmax and accepts float32 `N×3×224×224`, returning `N×12` probabilities. ONNX Runtime CPU results must agree with PyTorch on real validation images for batch sizes 1 and 8 before submission metadata is generated. The ONNX file checksum binds the model to its evaluation record.

No numerical result is asserted in this source document. The runner produces `shuaa-output/measured-report.md` from executed experiments, including completed epochs and stop reasons. Final test metrics come from the separate benchmark workflow.

## Reproduce

```bash
python -m pip install -r requirements-train.txt
python -c "from benchmark.dataset import prepare; prepare('.')"
python -m training.train --config configs/efficientnetv2_ce.json --dataset data/infrared --split benchmark/split.json --output runs/efficientnetv2_ce
python -m training.campaign --dataset data/infrared --split benchmark/split.json --output runs --budget-hours 11
```

Use `--resume` with the single-run training command to resume its checkpoint. The campaign resumes automatically. To extend a run beyond its configured budget, design a new registered experiment; do not silently change a checkpoint's configuration. Trusted checkpoints use PyTorch serialization and must come from your own training output.

For Kaggle, pin the reviewed source commit in `notebooks/shuaa_training.ipynb`, use `kernel-metadata.template.json`, enable internet and GPU, and execute. To resume after interruption, attach the prior output and set `SHUAA_PREVIOUS_OUTPUT` to the directory containing its `runs` folder. GPU use, authentication and Release publication are managed outside the notebook; it contains no secrets.
