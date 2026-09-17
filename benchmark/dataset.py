"""Pinned source, decoded-pixel duplicate grouping, immutable split manifests."""
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from .constants import CLASS_NAMES, DATASET_COMMIT, DATASET_SHA256, DATASET_URL


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "split_hash"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def decoded_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink rejected")
            if info.file_size > 100_000_000:
                raise ValueError("Unexpected archive member size")
        if sum(i.file_size for i in z.infolist()) > 2_000_000_000:
            raise ValueError("Unexpected archive size")
        z.extractall(destination)


def grouped_split(rows, seed=42):
    """Greedy stratification by label; exact duplicate groups never cross splits.

    Conflicting labels for identical pixels are rejected instead of silently assigned.
    Group indivisibility means proportions are approximate for duplicate-heavy data.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    conflicts = [key for key, items in groups.items() if len({r["label"] for r in items}) > 1]
    if conflicts:
        raise ValueError(f"Conflicting duplicate labels: {len(conflicts)} groups")
    by_label = defaultdict(list)
    for key, items in groups.items():
        by_label[items[0]["label"]].append(key)
    assignments = {}
    for label, keys in sorted(by_label.items()):
        keys = sorted(keys)
        random.Random(seed + label).shuffle(keys)
        # Largest groups first, seeded shuffle provides stable tie breaking.
        keys.sort(key=lambda key: len(groups[key]), reverse=True)
        total = sum(len(groups[key]) for key in keys)
        targets = [total * .70, total * .15, total * .15]
        counts = [0, 0, 0]
        for key in keys:
            chosen = max(range(3), key=lambda index: targets[index] - counts[index])
            counts[chosen] += len(groups[key])
            assignments[key] = ["train", "validation", "test"][chosen]
    return [dict(row, split=assignments[row["group"]]) for row in sorted(rows, key=lambda r: r["id"])]


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("classes") != CLASS_NAMES:
        raise ValueError("Unsupported manifest schema/classes")
    if manifest.get("split_hash") != manifest_hash(manifest):
        raise ValueError("Split manifest hash mismatch")
    seen, paths, groups = set(), set(), {}
    for row in manifest["samples"]:
        if row["id"] in seen or row["path"] in paths or row["split"] not in ("train", "validation", "test") or not 0 <= row["label"] < 12:
            raise ValueError("Invalid or duplicate manifest sample")
        if PurePosixPath(row["path"]).is_absolute() or ".." in PurePosixPath(row["path"]).parts or "\\" in row["path"] or ":" in row["path"]:
            raise ValueError("Unsafe sample path")
        if row["group"] in groups and groups[row["group"]] != (row["split"], row["label"]):
            raise ValueError("Duplicate leakage or conflicting labels")
        seen.add(row["id"])
        paths.add(row["path"])
        groups[row["group"]] = (row["split"], row["label"])
    for split in ("train", "validation", "test"):
        if {row["label"] for row in manifest["samples"] if row["split"] == split} != set(range(12)):
            raise ValueError("Each partition must contain all twelve classes")
    return manifest


def prepare(root):
    root = Path(root)
    data = root / "data"
    data.mkdir(exist_ok=True)
    archive = data / "source.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(DATASET_URL, temporary)
        if file_sha256(temporary) != DATASET_SHA256:
            temporary.unlink()
            raise ValueError("Downloaded dataset checksum mismatch")
        temporary.replace(archive)
    if file_sha256(archive) != DATASET_SHA256:
        raise ValueError("Dataset archive checksum mismatch")
    canonical = root / "benchmark" / "split.json"
    target = data / "infrared"
    if canonical.exists() and (target / "images").is_dir():
        existing = load_manifest(canonical)
        if all((target / row["path"]).is_file() for row in existing["samples"]):
            for row in existing["samples"]:
                if decoded_hash(target / row["path"]) != row["group"]:
                    raise ValueError("Local image differs from committed benchmark")
            (data / "split.json").write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
            return existing
    extracted = data / "source"
    if not (extracted / "InfraredSolarModules" / "module_metadata.json").exists() or len(list((extracted / "InfraredSolarModules" / "images").glob("*.jpg"))) != 20_000:
        safe_extract(archive, extracted)
    source = extracted / "InfraredSolarModules"
    if not target.exists():
        source.rename(target)
        source = target
    else:
        (target / "images").mkdir(exist_ok=True)
        for source_image in (source / "images").glob("*.jpg"):
            target_image = target / "images" / source_image.name
            if not target_image.exists():
                shutil.copy2(source_image, target_image)
    metadata = json.loads((source / "module_metadata.json").read_text())
    rows = []
    for sample_id, item in sorted(metadata.items()):
        image_path = item["image_filepath"]
        if not image_path.startswith("images/") or ".." in PurePosixPath(image_path).parts:
            raise ValueError("Unexpected metadata image path")
        rows.append({"id": sample_id, "path": image_path, "label": CLASS_NAMES.index(item["anomaly_class"]), "group": decoded_hash(target / image_path)})
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    conflicts = [{"group": key, "ids": [r["id"] for r in value], "labels": sorted({r["label"] for r in value})} for key, value in groups.items() if len({r["label"] for r in value}) > 1]
    audit = {"total_images": len(rows), "unique_pixel_groups": len(groups), "duplicate_groups": sum(len(v) > 1 for v in groups.values()), "duplicate_extra_images": len(rows) - len(groups), "conflicting_groups": conflicts, "near_duplicates_checked": False}
    (data / "duplicate-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Contradictory annotations have no defensible single target. Quarantine the
    # entire pixel group before splitting, without inspecting model performance.
    conflicting_hashes = {item["group"] for item in conflicts}
    excluded = [row for row in rows if row["group"] in conflicting_hashes]
    eligible = [row for row in rows if row["group"] not in conflicting_hashes]
    audit["conflict_policy"] = "exclude_all_images_in_conflicting_pixel_groups"
    audit["excluded_image_ids"] = [row["id"] for row in excluded]
    audit["eligible_images"] = len(eligible)
    samples = grouped_split(eligible)
    audit["split_counts"] = dict(Counter(r["split"] for r in samples))
    (data / "duplicate-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "seed": 42, "classes": CLASS_NAMES, "dataset": {"commit": DATASET_COMMIT, "url": DATASET_URL, "sha256": DATASET_SHA256}, "audit": audit, "samples": samples}
    manifest["split_hash"] = manifest_hash(manifest)
    if canonical.exists() and load_manifest(canonical)["split_hash"] != manifest["split_hash"]:
        raise ValueError("Dataset differs from immutable committed split")
    serialized = json.dumps(manifest, separators=(",", ":")) + "\n"
    canonical.write_text(serialized, encoding="utf-8")
    (data / "split.json").write_text(serialized, encoding="utf-8")
    return manifest
