"""Lock validation submissions before the one-time final test."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.dataset import file_sha256, load_manifest
from benchmark.evaluation import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submissions', nargs='+', help='Submission JSON paths, one per person')
    parser.add_argument('--output', type=Path, default=ROOT / 'config/finalists.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Finalists already locked. Do not overwrite a lock after viewing test results.')
    manifest = load_manifest(ROOT / 'benchmark/split.json')
    finalists = []
    for filename in args.submissions:
        path = Path(filename)
        submission = validate_submission(json.loads(path.read_text(encoding='utf-8')), manifest['split_hash'])
        if path.stem != submission['id']:
            parser.error('Submission filename must equal its id')
        finalists.append({'participant': submission['participant'], 'submission_id': submission['id'],
                          'submission_sha256': file_sha256(path), 'checkpoint_sha256': submission['checkpoint_sha256']})
    if len({entry['participant'] for entry in finalists}) != len(finalists):
        parser.error('Choose exactly one model per participant')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema_version': 1, 'locked': True, 'finalists': finalists}, indent=2) + '\n', encoding='utf-8')
    print(f'Locked {len(finalists)} finalist(s): {args.output}')


if __name__ == '__main__':
    main()
