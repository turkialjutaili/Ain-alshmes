"""Download verified source and reproduce the committed benchmark split."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.dataset import prepare

if __name__ == "__main__":
    manifest = prepare(ROOT)
    print(json.dumps({"split_hash": manifest["split_hash"], "audit": manifest["audit"]}, indent=2))
