#!/usr/bin/env python3
"""
test_watcher.py - Unit Tests for DataWatcher Daemon and nlp-fuzzy-expert package init.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLP_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")
if NLP_DIR not in sys.path:
    sys.path.insert(0, NLP_DIR)

import watcher
from watcher import DataWatcher


class TestDataWatcher(unittest.TestCase):
    """Verifies DataWatcher snapshot detection, change tracking, daemon loop, CLI, and package init."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.sample_file = os.path.join(self.data_dir, "sample.json")
        with open(self.sample_file, "w") as f:
            f.write('{"key": "initial"}')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_package_init_imports(self):
        """Verifies nlp-fuzzy-expert/__init__.py exports all core symbols."""
        init_path = os.path.join(NLP_DIR, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            "nlp_fuzzy_expert_pkg", init_path, submodule_search_locations=[NLP_DIR]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["nlp_fuzzy_expert_pkg"] = pkg
        spec.loader.exec_module(pkg)
        self.assertIn("ExpertSynchronizer", pkg.__all__)
        self.assertIn("KnowledgeBase", pkg.__all__)

    def test_get_data_files_and_snapshot(self):
        """Verifies retrieving data files and taking snapshots (including missing dir)."""
        with patch("watcher.DATA_DIR", self.data_dir), patch("watcher.ExpertSynchronizer"):
            w = DataWatcher(poll_interval=0.1)
            w.synchronizer = MagicMock()
            files = w.get_data_files()
            self.assertEqual(len(files), 1)
            self.assertIn(self.sample_file, w.file_mtimes)

        with patch("watcher.DATA_DIR", os.path.join(self.temp_dir, "nonexistent")), patch("watcher.ExpertSynchronizer"):
            w2 = DataWatcher(poll_interval=0.1)
            self.assertEqual(w2.get_data_files(), [])

    def test_check_for_changes_modified_created_deleted(self):
        """Verifies detecting created, modified, and deleted data files."""
        with patch("watcher.DATA_DIR", self.data_dir), patch("watcher.ExpertSynchronizer"):
            w = DataWatcher(poll_interval=0.1)
            w.synchronizer = MagicMock()

            self.assertEqual(len(w.check_for_changes()), 0)

            with open(self.sample_file, "w") as f:
                f.write('{"key": "modified_content_longer"}')
            changes = w.check_for_changes()
            self.assertGreaterEqual(len(changes), 1)
            self.assertEqual(changes[0][1], "modified")

            w.update_snapshot()

            new_file = os.path.join(self.data_dir, "new.json")
            with open(new_file, "w") as f:
                f.write('{"new": 1}')
            changes = w.check_for_changes()
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0][1], "created")

            w.update_snapshot()

            os.remove(new_file)
            changes = w.check_for_changes()
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0][1], "deleted")

    def test_handle_signal(self):
        """Verifies handle_signal sets running flag to False."""
        with patch("watcher.ExpertSynchronizer"), patch("sys.stdout", new_callable=io.StringIO):
            w = DataWatcher(poll_interval=0.1)
            w.synchronizer = MagicMock()
            self.assertTrue(w.running)
            w.handle_signal(2, None)
            self.assertFalse(w.running)

    def test_default_poll_interval(self):
        """Verifies default poll interval is 60.0 seconds."""
        with patch("watcher.ExpertSynchronizer"):
            w = DataWatcher()
            w.synchronizer = MagicMock()
            self.assertEqual(w.poll_interval, 60.0)

    def test_watcher_run_loop_and_main_cli(self):
        """Verifies DataWatcher.run() polling cycle, error recovery, and main() CLI modes."""
        with patch("watcher.DATA_DIR", self.data_dir), \
             patch("watcher.REPO_DIR", self.temp_dir), \
             patch("watcher.ExpertSynchronizer") as mock_sync_cls, \
             patch("signal.signal"), \
             patch("sys.stdout", new_callable=io.StringIO), \
             patch("sys.stderr", new_callable=io.StringIO):
            w = DataWatcher(poll_interval=0.01)
            step = {"n": 0}

            def fake_sleep(_):
                step["n"] += 1
                if step["n"] == 1:
                    # Modify sample file to trigger sync
                    with open(self.sample_file, "w") as f:
                        f.write('{"key": "updated_in_loop"}')
                elif step["n"] == 2:
                    # Debounce sleep
                    pass
                elif step["n"] == 3:
                    # Trigger an exception in check_for_changes to test error recovery
                    raise RuntimeError("simulated polling error")
                else:
                    w.running = False

            with patch("time.sleep", side_effect=fake_sleep):
                w.run()
            self.assertGreaterEqual(w.synchronizer.sync_all.call_count, 2)

            # Test main(--once)
            with patch("sys.argv", ["watcher.py", "--once"]):
                watcher.main()
            self.assertTrue(mock_sync_cls.called)

            # Test main() daemon mode
            with patch("sys.argv", ["watcher.py", "--interval", "5.0"]), \
                 patch.object(DataWatcher, "run") as mock_run:
                watcher.main()
                mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
