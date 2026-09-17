# Submit a model to Shuaa

## 1. Use the shared data split

Run `python scripts/prepare_dataset.py`. Train only on the `train` samples in `benchmark/split.json`, select settings with `validation`, and leave `test` unused until final lock. The order in `benchmark.constants.CLASS_NAMES` is the only supported output order.

## 2. Export an ONNX probability model

Use `python -m training.export --help` for the supplied PyTorch exporter. Models from other frameworks can participate if they satisfy the same contract:

- One float32 input: `[batch, 3, height, width]`, with a dynamic batch dimension.
- Decode each original image to RGB, resize the **whole** image with Pillow bilinear interpolation to the declared square input size, divide by 255, then normalize by ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- One output: `[batch, 12]` probabilities, finite and nonnegative, summing to one. Softmax belongs inside the exported model.
- No external ONNX weight files. Compare the ONNX probabilities with the original model on real validation images before publishing.

## 3. Upload weights and add a submission

Upload `.onnx` weights as an asset in a [Shuaa GitHub Release](https://github.com/turkialjutaili/Shuaa/releases). The checkpoint URL must point to that repository; record its SHA-256. Use a new release asset name for each model version.

Copy `submissions/example.json.example` to `submissions/<unique-id>.json` and replace its values. The JSON schema is `benchmark/submission.schema.json`. Participant IDs are `turki`, `muhannad`, and `mazen`. Include the architecture paper, exact training code commit, checkpoint hash, split hash and preprocessing; do not supply accuracy yourself.

Push the submission or merge a pull request into `main`. Open the **Shuaa benchmark and Pages** workflow under Actions. It computes validation metrics, preserves the result in `results/`, regenerates the leaderboard and publishes the site. Failures are shown without replacing previous successes.

## 4. Lock the final test

Agree on one final submission per participating person before inspecting any test result. Use `python scripts/lock_finalists.py --help` to create the lock file after the validation stage. The lock includes the exact submission-file and checkpoint hashes.

The resulting `config/finalists.json` has this shape:

```json
{
  "schema_version": 1,
  "locked": true,
  "finalists": [
    {
      "participant": "turki",
      "submission_id": "turki-selected-model",
      "submission_sha256": "<SHA-256 of the exact submission JSON file>",
      "checkpoint_sha256": "<SHA-256 of the ONNX file>"
    }
  ]
}
```

Commit the lock and run the workflow manually with `phase: test`. A changed finalist is rejected; successful identical evaluations are reused. Do not change the lock after seeing test results and claim the revised result was an independent holdout.

## Trust and limits

This is a cooperative three-person benchmark. Repository writers can change code, and original labels are public. It is not an adversarial judging service. It verifies common inference and metrics, not whether a participant secretly trained on test data.

CPU inference has a finite workflow time limit. Invalid graphs, missing assets, mismatched hashes or invalid probabilities are failures, not zero-accuracy models. The previous published leaderboard remains available when infrastructure fails.
