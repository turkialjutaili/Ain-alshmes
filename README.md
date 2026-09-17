# Shuaa | شعاع

**A reproducible solar thermal model competition for Turki, Muhannad and Mazen.**

[Competition website](https://turkialjutaili.github.io/Shuaa/) · [How to submit](docs/SUBMISSIONS.md) · [العربية](docs/README.ar.md)

Shuaa compares classifiers on the original 12-class [Raptor Maps InfraredSolarModules dataset](https://github.com/RaptorMaps/InfraredSolarModules). The website uses English by default and offers Arabic with right-to-left layout. Results are produced by the shared evaluator, never entered as claimed accuracy.

## Competition protocol

- One fixed, versioned 70/15/15 train/validation/test split, seed 42. Exact decoded-pixel duplicates stay together. The committed manifest records exact counts and its content hash.
- The source contains 20,000 images. Six identical-pixel pairs have conflicting labels; all 12 images are quarantined before splitting, leaving 19,988 eligible images. Excluded IDs and the policy are recorded in the manifest audit. Near-duplicate and farm-level independence are not established.
- All repeated model development and the live leaderboard use **validation**. A separate manually triggered final evaluates one locked model per participant on **test**.
- Rank by accuracy, then macro-F1. Show per-class precision/recall/F1, balanced accuracy and the confusion matrix as well. A complete tie shares a place.
- Train augmentations are never applied to validation/test. No oversampling changes the validation/test class distribution.
- Labels and source images are public: holding out test is a team research protocol, not a secret competition server. Farm/flight identifiers are not supplied, so this benchmark does not establish unseen-farm generalization.
- Failed entries remain visible; they do not remove previous successful results. Different checkpoint/configuration hashes are distinct evaluations.

## Run the website

Requires Node.js 22 and Python 3.11 for evaluation.

```sh
npm ci
npm run dev
npm run build
```

Only the generated `dist/` website is deployed. GitHub Pages is static hosting: it does not run training or visitor inference.

## Prepare the benchmark

```sh
python -m pip install -r requirements-eval.txt
python scripts/prepare_dataset.py
python scripts/evaluate_submissions.py --phase validation
```

The preparation command downloads the pinned original archive, checks SHA-256, verifies duplicate grouping, and checks the committed split. Images and checkpoints remain ignored under `data/`. No actual submitted model means no score on the leaderboard.

## Train paper-backed models

Training lives under `training/`; Kaggle notebooks are in `notebooks/`. Use a GPU runtime and install `requirements-train.txt` there. Do not install CUDA training packages on the Pages runner.

```sh
python -m training.train --config configs/efficientnetv2_ce.json --dataset data/infrared --split benchmark/split.json --output runs/efficientnetv2_ce
python -m training.campaign --dataset data/infrared --split benchmark/split.json --output runs --budget-hours 11
```

The campaign compares EfficientNetV2-S and ConvNeXtV2-Tiny with ordinary and weighted cross-entropy, repeats the best setting with another seed, and considers averaging the two best probability outputs if validation improves. A campaign can be partial when compute expires: inspect its status and resume from saved outputs. Do not label an incomplete comparison a global best model.

- [EfficientNetV2 — ICML 2021](https://proceedings.mlr.press/v139/tan21a.html)
- [ConvNeXt V2 — CVPR 2023](https://arxiv.org/abs/2301.00808)
- [Dataset paper — ICLR 2020 AI for Earth Sciences workshop](https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth22.pdf)

## Automation and final test

Pull requests build the site and test the benchmark. Merging to `main` evaluates model submissions, saves computed results and deploys Pages. Large weights are release assets; access credentials never belong in submissions, notebooks or the repository.

See [submission instructions](docs/SUBMISSIONS.md) for the ONNX contract, release upload, immutable finalist lock and final workflow dispatch. GitHub Actions can run inference on CPU; GPU training is a separate Kaggle workload.

## Verification

```sh
python -m pytest tests -q
npm run build
```

Training tests require `requirements-train.txt`; evaluation-only checks use the benchmark tests. Research notes and the honest results report are under `research/`.
