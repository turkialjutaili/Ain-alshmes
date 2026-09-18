"""Independent CPU inference and metrics; submissions are data, never Python code."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from .constants import CLASS_NAMES, MEAN, STD, PARTICIPANTS
from .dataset import decoded_hash, file_sha256


def validate_submission(submission, split_hash):
    import jsonschema
    schema = json.loads((Path(__file__).parent / "submission.schema.json").read_text())
    jsonschema.validate(submission, schema)
    if submission["split_hash"] != split_hash:
        raise ValueError("Submission uses a different benchmark split")
    if submission["classes"] != CLASS_NAMES:
        raise ValueError("Class order mismatch")
    if submission["preprocess"]["mean"] != MEAN or submission["preprocess"]["std"] != STD:
        raise ValueError("Only benchmark ImageNet normalization is supported")
    return submission


def validate_probabilities(value, count):
    value = np.asarray(value)
    if value.shape != (count, len(CLASS_NAMES)) or value.dtype != np.float32:
        raise ValueError("Output must be float32 probabilities with shape (N,12)")
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.allclose(value.sum(axis=1), 1, rtol=1e-4, atol=1e-5):
        raise ValueError("Output rows must sum to one; include softmax in the exported graph")
    return value


def metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("Metrics require equal, nonempty one-dimensional label arrays")
    if not np.isin(labels, range(12)).all() or not np.isin(predictions, range(12)).all():
        raise ValueError("Invalid class index")
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(12), zero_division=0)
    return {"accuracy": float(np.mean(labels == predictions)), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[support > 0].mean()), "per_class": [{"label": label, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(CLASS_NAMES)], "confusion_matrix": confusion_matrix(labels, predictions, labels=range(12)).tolist()}


def preprocess_image(path, size):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    return ((array - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)).transpose(2, 0, 1)


def checkpoint(submission, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (submission["checkpoint_sha256"] + ".onnx")
    if not target.exists():
        temporary = target.with_suffix(".download")
        with urllib.request.urlopen(submission["checkpoint_url"], timeout=60) as response, temporary.open("wb") as output:
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > 2_000_000_000:
                    raise ValueError("Checkpoint exceeds 2GB limit")
                output.write(block)
        if file_sha256(temporary) != submission["checkpoint_sha256"]:
            temporary.unlink()
            raise ValueError("Checkpoint checksum mismatch")
        temporary.replace(target)
    if file_sha256(target) != submission["checkpoint_sha256"]:
        raise ValueError("Cached checkpoint checksum mismatch")
    return target


def infer(submission, manifest, phase, image_root, checkpoint_path, batch_size=32):
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(checkpoint_path), load_external_data=False)
    def reject_external(proto):
        if isinstance(proto, onnx.TensorProto) and (proto.external_data or proto.data_location == onnx.TensorProto.EXTERNAL):
            raise ValueError("External ONNX tensor files are not allowed")
        for descriptor, value in proto.ListFields():
            if descriptor.type == descriptor.TYPE_MESSAGE:
                repeated = descriptor.is_repeated if hasattr(descriptor, "is_repeated") else descriptor.label == descriptor.LABEL_REPEATED
                if repeated:
                    for child in value:
                        reject_external(child)
                else:
                    reject_external(value)
    reject_external(graph)
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(checkpoint_path), sess_options=opts, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    size = submission["preprocess"]["input_size"]
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4 or inputs[0].shape[1:] != [3, size, size]:
        raise ValueError("ONNX input must be float32 NCHW with declared image size")
    if isinstance(inputs[0].shape[0], int):
        raise ValueError("ONNX input must have a dynamic batch dimension")
    if len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 12:
        raise ValueError("ONNX must have exactly one float32 output of shape N,12")
    rows = [row for row in manifest["samples"] if row["split"] == phase]
    if not rows or {row["label"] for row in rows} != set(range(12)):
        raise ValueError("Evaluation partition must include all twelve classes")
    labels, predicted = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        arrays = []
        for row in batch:
            path = Path(image_root) / row["path"]
            if decoded_hash(path) != row["group"]:
                raise ValueError("Image pixel hash differs from canonical benchmark")
            arrays.append(preprocess_image(path, size))
        tensor = np.stack(arrays).astype(np.float32)
        probabilities = validate_probabilities(session.run(None, {inputs[0].name: tensor})[0], len(batch))
        labels.extend(row["label"] for row in batch)
        predicted.extend(probabilities.argmax(axis=1).tolist())
    return dict(metrics(labels, predicted), sample_count=len(labels), predictions=[{"id": row["id"], "label": label, "predicted": prediction} for row, label, prediction in zip(rows, labels, predicted)])


def evaluation_id(submission, phase):
    # Training notes can be enriched after a measurement. Every other manifest
    # field, including identity and preprocessing, remains bound to the result.
    immutable_submission = {key: value for key, value in submission.items() if key != "training"}
    return hashlib.sha256(json.dumps({"submission": immutable_submission, "phase": phase, "evaluator_version": 1}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_leaderboard(root):
    root = Path(root)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "results").glob("*.json"))]
    results.sort(key=lambda r: (r.get("evaluated_at", ""), r["id"]), reverse=True)
    payload = {"schema_version": 1, "updated_at": max((r["evaluated_at"] for r in results), default=None), "classes": CLASS_NAMES, "participants": PARTICIPANTS, "results": results}
    target = root / "public" / "data" / "leaderboard.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def now():
    return datetime.now(timezone.utc).isoformat()
