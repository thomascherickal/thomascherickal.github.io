#!/usr/bin/env python3
"""
build.py - Backward-compatible wrapper for build_components.py.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_components import build, watch

if __name__ == "__main__":
    if "--watch" in sys.argv:
        build()
        watch()
    else:
        build()
