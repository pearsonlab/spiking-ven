# Notes for Claude Code

## Checks after every code change

Whenever you modify code (package, scripts or tests), run both checks CI runs and fix
everything they report before committing:

```
uv run --all-extras ruff check .
uv run --all-extras ty check
```

`ty` is blocking in CI (`.github/workflows/ci.yml`); do not leave type errors for CI to
find. Fix the cause (annotate, narrow `Optional`s, type `**kwargs` dicts as
`dict[str, Any]`) rather than adding `# type: ignore`. `ty` also flags *unused*
`type: ignore` comments, so remove those when they stop being needed.

If `uv run` cannot reach PyPI (TLS interception), call the tools from an existing venv
that has the `dev` extra: `.venv/bin/ruff check .` and
`.venv/bin/ty check --python .venv` (from a worktree, point `--python` at the main
checkout's `.venv`).
