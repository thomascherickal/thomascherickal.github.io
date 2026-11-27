#!/usr/bin/env python3
"""
export_site_data.py - Generate assets/data/site-data.json from data/*.json for JS.
Never edit assets/data/site-data.json by hand.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.ssot.load import SSOTData


def export_site_data(out_path=None):
    if out_path is None:
        out_path = os.path.join(REPO_ROOT, "assets", "data", "site-data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = SSOTData()
    payload = {
        "pricing": data.raw["pricing"],
        "tiers": data.raw["tiers"],
        "contact": data.raw["contact"],
        "config": data.raw["config"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path


if __name__ == "__main__":
    p = export_site_data()
    print(f"Generated {os.path.relpath(p, REPO_ROOT)}")
