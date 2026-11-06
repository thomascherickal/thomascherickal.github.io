#!/usr/bin/env python3
"""
test_watcher.py - Unit Tests for DataWatcher Daemon.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLP_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
sys.path.insert(0, NLP_DIR)

from watcher import DataWatcher


class TestDataWatcher(unittest.TestCase):
    """Verifies DataWatcher snapshot detection, change tracking, and signal handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        # Create a sample data file
        self.sample_file = os.path.join(self.data_dir, "sample.json")
        with open(self.sample_file, "w") as f:
            f.write('{"key": "initial"}')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_data_files_and_snapshot(self):
        """Verifies retrieving data files and taking snapshots."""
        with patch("watcher.DATA_DIR", self.data_dir):
            with patch("watcher.ExpertSynchronizer"):
                watcher = DataWatcher(poll_interval=0.1)
                watcher.synchronizer = MagicMock()
                files = watcher.get_data_files()
                self.assertEqual(len(files), 1)
                self.assertIn(self.sample_file, watcher.file_mtimes)

    def test_check_for_changes_modified_created_deleted(self):
        """Verifies detecting created, modified, and deleted data files."""
        with patch("watcher.DATA_DIR", self.data_dir):
            with patch("watcher.ExpertSynchronizer"):
                watcher = DataWatcher(poll_interval=0.1)
                watcher.synchronizer = MagicMock()

                # No changes initially
                changes = watcher.check_for_changes()
                self.assertEqual(len(changes), 0)

                # Modify existing file
                with open(self.sample_file, "w") as f:
                    f.write('{"key": "modified_content_longer"}')
                changes = watcher.check_for_changes()
                self.assertGreaterEqual(len(changes), 1)
                self.assertEqual(changes[0][1], "modified")

                watcher.update_snapshot()

                # Create a new file
                new_file = os.path.join(self.data_dir, "new.json")
                with open(new_file, "w") as f:
                    f.write('{"new": 1}')
                changes = watcher.check_for_changes()
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0][1], "created")

                watcher.update_snapshot()

                # Delete the new file
                os.remove(new_file)
                changes = watcher.check_for_changes()
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0][1], "deleted")

    def test_handle_signal(self):
        """Verifies handle_signal sets running flag to False."""
        with patch("watcher.ExpertSynchronizer"):
            watcher = DataWatcher(poll_interval=0.1)
            watcher.synchronizer = MagicMock()
            self.assertTrue(watcher.running)
            watcher.handle_signal(2, None)
            self.assertFalse(watcher.running)


if __name__ == "__main__":
    unittest.main()
