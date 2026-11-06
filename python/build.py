#!/usr/bin/env python3
"""
build.py - Backward-compatible wrapper for build_components.py.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_components import build, watch


def main(argv=None):
    args = argv if argv is not None else sys.argv
    if "--watch" in args:
        build()
        watch()
    else:
        build()


if __name__ == "__main__":  # pragma: no cover
    main()
