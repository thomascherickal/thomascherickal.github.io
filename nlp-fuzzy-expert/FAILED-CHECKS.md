# Failed Checks Report

- **Status**: 1 failure(s) detected

## Details

```text
FAIL: test_claude_guides_completeness (test_sync_claude_and_gemini.TestSyncClaudeAndGemini.test_claude_guides_completeness)
Traceback (most recent call last):
  File "/home/thomas/GitHub.io/thomascherickal.github.io/tests/test_sync_claude_and_gemini.py", line 80, in test_claude_guides_completeness
    self.assertTrue(os.path.exists(path), f"Claude guide {slug} must exist on disk")
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: False is not true : Claude guide claude-features.html must exist on disk
```
