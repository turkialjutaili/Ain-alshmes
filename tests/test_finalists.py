"""The final holdout cannot silently run on a changed or unlocked model."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.dataset import file_sha256
from scripts.evaluate_submissions import select_submissions


class FinalistTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.submission = self.root / 'turki-model.json'
        self.submission.write_text(json.dumps({'id': 'turki-model', 'participant': 'turki', 'checkpoint_sha256': 'a' * 64}))
        self.lock = self.root / 'finalists.json'
        self.config = {'schema_version': 1, 'locked': True, 'finalists': [
            {'participant': 'turki', 'submission_id': 'turki-model',
             'checkpoint_sha256': 'a' * 64, 'submission_sha256': file_sha256(self.submission)}]}
        self.write_lock()

    def tearDown(self):
        self.directory.cleanup()

    def write_lock(self):
        self.lock.write_text(json.dumps(self.config))

    def test_final_requires_explicit_lock(self):
        with self.assertRaises(ValueError):
            select_submissions([self.submission], 'test', None)

    def test_unchanged_locked_model_is_selected(self):
        self.assertEqual(select_submissions([self.submission], 'test', self.lock), [self.submission])

    def test_any_edit_after_lock_rejected(self):
        self.submission.write_text(self.submission.read_text() + '\n')
        with self.assertRaises(ValueError):
            select_submissions([self.submission], 'test', self.lock)

    def test_two_models_for_same_person_rejected(self):
        self.config['finalists'].append(dict(self.config['finalists'][0]))
        self.write_lock()
        with self.assertRaises(ValueError):
            select_submissions([self.submission], 'test', self.lock)

    def test_validation_does_not_use_final_lock(self):
        with self.assertRaises(ValueError):
            select_submissions([self.submission], 'validation', self.lock)


if __name__ == '__main__':
    unittest.main()
