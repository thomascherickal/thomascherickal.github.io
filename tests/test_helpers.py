"""Shared modular helpers for the repository unit test suite."""

import glob
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")
NLP_EXPERT_DIR = os.path.join(REPO_DIR, "nlp-fuzzy-expert")


def load_json(path):
    """Load and parse a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config_json_count():
    """Return the expected data/*.json file count from data/config.json."""
    return load_json(os.path.join(DATA_DIR, "config.json"))["count-json-files"]


def get_disk_data_json_files():
    """Return all .json file paths currently present in data/."""
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

