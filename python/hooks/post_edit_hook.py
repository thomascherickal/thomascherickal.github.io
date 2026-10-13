#!/usr/bin/env python3
"""
post_edit_hook.py — Lifecycle hook triggered after file edits.
If header.html or footer.html was edited, automatically runs build_components.py.
"""
import json
import os
import subprocess
import sys

def main():
    try:
        raw_input = sys.stdin.read()
        if raw_input.strip():
            data = json.loads(raw_input)
            tool_args = data.get("toolCall", {}).get("args", {})
            target_file = tool_args.get("TargetFile", "")
            if "header.html" in target_file or "footer.html" in target_file:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                build_script = os.path.join(repo_root, "python", "build_components.py")
                if os.path.exists(build_script):
                    subprocess.run(
                        [sys.executable, build_script],
                        cwd=repo_root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
    except Exception:
        pass
    finally:
        # PostToolUse contract requires a JSON object on stdout
        print("{}")

if __name__ == "__main__":
    main()
