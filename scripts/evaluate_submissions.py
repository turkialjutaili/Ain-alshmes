"""Evaluate validation submissions or an explicitly locked set of test finalists."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.constants import CLASS_NAMES
from benchmark.dataset import file_sha256, load_manifest
from benchmark.evaluation import checkpoint, evaluation_id, infer, now, validate_submission, write_leaderboard


def select_submissions(paths, phase, finalists_path):
    if phase == "validation":
        if finalists_path:
            raise ValueError("Finalists only apply to test evaluation")
        return paths
    if not finalists_path:
        raise ValueError("Test evaluation requires --finalists with locked submission hashes")
    config = json.loads(Path(finalists_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("locked") is not True:
        raise ValueError("Finalists config must have schema_version=1 and locked=true")
    finalists = config["finalists"]
    if not finalists or len({r["participant"] for r in finalists}) != len(finalists):
        raise ValueError("Exactly one finalist per participating person")
    selected = []
    for item in finalists:
        matches = [path for path in paths if path.stem == item["submission_id"]]
        if len(matches) != 1:
            raise ValueError("Finalist submission missing or ambiguous")
        path = matches[0]
        submission = json.loads(path.read_text(encoding="utf-8"))
        if file_sha256(path) != item["submission_sha256"] or submission["checkpoint_sha256"] != item["checkpoint_sha256"] or submission["participant"] != item["participant"]:
            raise ValueError("Finalist changed after lock")
        selected.append(path)
    return selected


def cached_success(root, submission, phase, evaluation_key):
    """Return only the exact manifest's prior successful measurement."""
    path = root / "results" / f"{evaluation_key}.json"
    if not path.exists():
        return None, None
    record = json.loads(path.read_text(encoding="utf-8"))
    bound = ("id", "participant", "model_name", "source_commit", "paper_url", "code_url", "checkpoint_sha256", "split_hash")
    if (record.get("status") != "success" or record.get("phase") != phase
            or any(record.get(key) != submission[key] for key in bound)):
        return None, None
    # Existing files are named by the immutable submission evaluation ID. New
    # records also carry this field, while the filename preserves legacy proof.
    if path.stem != evaluation_key:
        return None, None
    immutable_submission = {key: value for key, value in submission.items() if key != "training"}
    # New records preserve the complete inference contract. A legacy record can
    # be trusted only when its original evaluation hash proves the same contract.
    if "submission_contract" in record:
        if record["submission_contract"] != immutable_submission:
            return None, None
    elif record.get("evaluation_id") != evaluation_key:
        return None, None
    return path, record


def enrich_cached_result(record, submission, submission_sha256, evaluation_key):
    """Attach current, validated provenance without changing measured metrics."""
    record.update({key: submission[key] for key in ["id", "participant", "model_name", "source_commit", "paper_url", "code_url", "checkpoint_sha256"]})
    if "training" in submission:
        record["training"] = submission["training"]
    else:
        record.pop("training", None)
    record["submission_sha256"] = submission_sha256
    record["evaluation_id"] = evaluation_key
    record["submission_contract"] = {key: value for key, value in submission.items() if key != "training"}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "test"], default="validation")
    parser.add_argument("--finalists", type=Path)
    parser.add_argument("--submission", type=Path, action="append")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128:
        parser.error("batch-size must be 1..128")
    paths = args.submission or sorted((ROOT / "submissions").glob("*.json"))
    paths = select_submissions(paths, args.phase, args.finalists)
    (ROOT / "results").mkdir(exist_ok=True)
    if not paths:
        write_leaderboard(ROOT)
        print("No submissions; leaderboard contains no invented scores.")
        return 0
    manifest = load_manifest(ROOT / "benchmark" / "split.json")
    failed = 0
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        # Parse failures are recorded under a safe ID without trusting metadata.
        base = {"id": path.stem, "participant": "unknown", "model_name": path.stem, "phase": args.phase, "evaluated_at": now(), "split_hash": manifest["split_hash"]}
        result_key = __import__("hashlib").sha256((raw + args.phase).encode()).hexdigest()
        try:
            submission = validate_submission(json.loads(raw), manifest["split_hash"])
            if path.stem != submission["id"]:
                raise ValueError("Submission filename must equal its id")
            result_key = evaluation_id(submission, args.phase)
            result_path = ROOT / "results" / f"{result_key}.json"
            cached_path, cached = cached_success(ROOT, submission, args.phase, result_key)
            if cached:
                enriched = enrich_cached_result(cached, submission, file_sha256(path), result_key)
                cached_path.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
                print(f"Cached successful evaluation metadata enriched: {submission['id']}")
                continue
            base.update({key: submission[key] for key in ["id", "participant", "model_name", "source_commit", "paper_url", "code_url", "checkpoint_sha256"]})
            if "training" in submission:
                base["training"] = submission["training"]
            base["submission_contract"] = {key: value for key, value in submission.items() if key != "training"}
            base["submission_sha256"] = file_sha256(path)
            model_path = checkpoint(submission, ROOT / "data" / "checkpoints")
            base.update(infer(submission, manifest, args.phase, ROOT / "data" / "infrared", model_path, args.batch_size))
            prediction_dir = ROOT / "results" / "predictions"
            prediction_dir.mkdir(exist_ok=True)
            prediction_path = prediction_dir / f"{result_key}.json"
            prediction_path.write_text(json.dumps(base.pop("predictions"), separators=(",", ":")) + "\n", encoding="utf-8")
            base["predictions_path"] = prediction_path.relative_to(ROOT).as_posix()
            base["status"] = "success"
        except Exception as exc:
            failed += 1
            base.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:800]}")
            # Preserve all prior failures and successful evaluations.
            result_key += "-failed-" + base["evaluated_at"].replace(":", "").replace(".", "")
            print(base["error"], file=sys.stderr)
        base["evaluation_id"] = result_key
        (ROOT / "results" / f"{result_key}.json").write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    write_leaderboard(ROOT)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
