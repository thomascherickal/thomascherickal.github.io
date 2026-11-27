#!/usr/bin/env python3
"""
load.py - Load and index data/*.json for SSOT rendering and formatting.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")


class SSOTData:
    """Indexed view over all data/*.json files."""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.raw = {}
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".json"):
                key = fname[:-5]
                with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                    self.raw[key] = json.load(f)

        self.offerings = {
            o["id"]: o for o in self.raw.get("pricing", {}).get("offerings", []) if "id" in o
        }
        self.tiers = {
            str(t["tier"]): t for t in self.raw.get("tiers", {}).get("tiers", []) if "tier" in t
        }
        self.tiers_list = self.raw.get("tiers", {}).get("tiers", [])

    def get(self, dotted_path):
        """Resolve a dotted path like 'offerings.flagship-personal-training.cohort_capacity.max_seats'."""
        parts = dotted_path.split(".")
        if not parts:
            raise KeyError("Empty SSOT path")

        head = parts[0]
        if head == "offerings":
            cur = self.offerings
        elif head == "tiers":
            cur = self.tiers
        elif head in self.raw:
            cur = self.raw[head]
        elif head in self.raw.get("pricing", {}):
            cur = self.raw["pricing"][head]
        elif head in self.raw.get("config", {}):
            cur = self.raw["config"][head]
        else:
            raise KeyError(f"Unknown SSOT root key: {head!r} in {dotted_path!r}")

        for part in parts[1:]:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
            else:
                raise KeyError(f"Missing SSOT path segment {part!r} in {dotted_path!r}")
        return cur
