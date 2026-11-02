#!/usr/bin/env python3
"""
watcher.py - Real-Time Autonomous Data Watcher Daemon for NLP-Fuzzy-Expert System.

Monitors data/*.json for changes using non-blocking filesystem polling,
and automatically triggers repository-wide Expert System forward chaining
synchronization upon modification.

Strict Repo Compliance:
- Zero emojis.
- Zero em-dashes (spaced hyphens ' - ' or plain hyphens '-' used).
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, SCRIPT_DIR)
from synchronizer import ExpertSynchronizer

class DataWatcher:
    """Monitors data/*.json and executes knowledge synchronization on changes."""

    def __init__(self, poll_interval=1.0):
        self.poll_interval = poll_interval
        self.synchronizer = ExpertSynchronizer()
        self.running = True
        self.file_mtimes = {}
        self.update_snapshot()

    def get_data_files(self):
        """Returns all JSON files in the data directory."""
        if not os.path.exists(DATA_DIR):
            return []
        return [
            os.path.join(DATA_DIR, fname)
            for fname in os.listdir(DATA_DIR)
            if fname.endswith(".json")
        ]

    def update_snapshot(self):
        """Takes a snapshot of current modification times and sizes."""
        snapshot = {}
        for fpath in self.get_data_files():
            try:
                stat = os.stat(fpath)
                snapshot[fpath] = (stat.st_mtime, stat.st_size)
            except OSError:
                continue
        self.file_mtimes = snapshot

    def check_for_changes(self):
        """Detects whether any data file has been created, modified, or deleted."""
        current_files = self.get_data_files()
        changed = []

        # Check modified or new
        for fpath in current_files:
            try:
                stat = os.stat(fpath)
                curr_state = (stat.st_mtime, stat.st_size)
                if fpath not in self.file_mtimes:
                    changed.append((fpath, "created"))
                elif self.file_mtimes[fpath] != curr_state:
                    changed.append((fpath, "modified"))
            except OSError:
                continue

        # Check deleted
        for old_fpath in self.file_mtimes:
            if not os.path.exists(old_fpath):
                changed.append((old_fpath, "deleted"))

        return changed

    def handle_signal(self, signum, frame):
        """Handles termination signals gracefully."""
        print(f"\n[NLP-Fuzzy-Expert Watcher] Received signal {signum}. Shutting down cleanly...", flush=True)
        self.running = False

    def run(self):
        """Starts the watcher daemon polling loop."""
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        print("[NLP-Fuzzy-Expert Watcher] Initializing repository synchronization...", flush=True)
        self.synchronizer.sync_all()
        self.update_snapshot()

        print(f"[NLP-Fuzzy-Expert Watcher] Daemon active. Monitoring {len(self.file_mtimes)} data files in data/...", flush=True)
        print(f"[NLP-Fuzzy-Expert Watcher] Polling interval: {self.poll_interval}s. Press Ctrl+C to terminate.", flush=True)

        while self.running:
            try:
                time.sleep(self.poll_interval)
                changes = self.check_for_changes()
                if changes:
                    names = ", ".join([os.path.basename(p) for p, _ in changes])
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{ts}] [NLP-Fuzzy-Expert Watcher] Change detected in data/ ({names}). Synchronizing...", flush=True)

                    # Debounce briefly to ensure write completion
                    time.sleep(0.15)
                    self.synchronizer.sync_all()
                    self.update_snapshot()
                    print(f"[{ts}] [NLP-Fuzzy-Expert Watcher] Synchronization complete. Standing by...", flush=True)
            except Exception as e:
                print(f"[NLP-Fuzzy-Expert Watcher] Error during sync: {e}", file=sys.stderr, flush=True)
                time.sleep(self.poll_interval)

        print("[NLP-Fuzzy-Expert Watcher] Daemon stopped.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="NLP-Fuzzy-Expert Data Watcher Daemon")
    parser.add_argument("--once", action="store_true", help="Run synchronization once and exit")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds (default: 1.0)")
    args = parser.parse_args()

    if args.once:
        print("[NLP-Fuzzy-Expert Watcher] Running single synchronization cycle...", flush=True)
        sync = ExpertSynchronizer()
        sync.sync_all()
        return

    watcher = DataWatcher(poll_interval=args.interval)
    watcher.run()

if __name__ == "__main__":
    main()

