#!/usr/bin/env python3
"""
post_edit_hook.py - Lifecycle hook triggered after file edits.
If header.html or footer.html was edited, automatically runs build_components.py.
"""
import json
import os
from pathlib import Path
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
                repo_root = Path(__file__).resolve().parents[2]
                build_script = repo_root / "python" / "build_components.py"
                if build_script.exists():
                    res = subprocess.run(
                        [sys.executable, str(build_script)],
                        cwd=str(repo_root),
                        capture_output=True,
                        text=True
                    )
                    if res.returncode != 0:
                        sys.stderr.write(f"build_components.py failed:\n{res.stderr}\n")
                else:
                    sys.stderr.write(f"build_script not found at: {build_script}\n")
    except Exception as e:
        sys.stderr.write(f"post_edit_hook error: {e}\n")
    finally:
        # PostToolUse contract requires a JSON object on stdout
        print("{}")

if __name__ == "__main__":
    main()
