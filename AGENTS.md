# AGENTS.md — prompt-complexity-analyzer

## Publishing a new version to PyPI

### Step 1 — Bump the version (two files, must match)

`pyproject.toml`:
```toml
version = "X.Y.Z"
```

`prompt_complexity_analyzer/__init__.py`:
```python
__version__ = "X.Y.Z"
```

### Step 2 — Commit and push

```bash
git add pyproject.toml prompt_complexity_analyzer/__init__.py
git commit -m "Bump version to X.Y.Z"
git push
```

### Step 3 — Tag the release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing a tag matching `v*` triggers `.github/workflows/publish.yml`, which builds and uploads to PyPI automatically via Trusted Publisher (OIDC — no API token required).

### Step 4 — Confirm

Check https://pypi.org/project/prompt-complexity-analyzer/ — new version appears within ~2 minutes.

---

## Key constraints

- Never re-upload an existing version — PyPI rejects it. Always increment before tagging.
- Keep `pyproject.toml` version and `__init__.__version__` in sync.
- The `pypi` GitHub environment must exist (Settings → Environments) for the workflow to run.
- PyPI Trusted Publisher must be configured at https://pypi.org/manage/project/prompt-complexity-analyzer/settings/publishing/ with owner `Ruthvik-Anne`, repo `prompt-complexity-analyzer`, workflow `publish.yml`, environment `pypi`.
