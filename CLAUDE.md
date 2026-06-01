# CLAUDE.md — prompt-complexity-analyzer

## Publish a new version to PyPI

1. **Bump the version** in two places (must match):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `prompt_complexity_analyzer/__init__.py` → `__version__ = "X.Y.Z"`

2. **Commit and push:**
   ```bash
   git add pyproject.toml prompt_complexity_analyzer/__init__.py
   git commit -m "Bump version to X.Y.Z"
   git push
   ```

3. **Tag the release** — this triggers the GitHub Actions publish workflow:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. **Verify** at https://pypi.org/project/prompt-complexity-analyzer/ — new version appears within ~2 minutes.

## Workflow location

`.github/workflows/publish.yml` — triggers on any tag matching `v*`, builds wheel + sdist, publishes via PyPI Trusted Publisher (no token needed).

## Package structure

| File | Purpose |
|---|---|
| `prompt_complexity_analyzer/_core.py` | All logic — keep in sync with `complexity_analyzer.py` |
| `prompt_complexity_analyzer/__init__.py` | Public API exports + auto-loads bundled model |
| `prompt_complexity_analyzer/model.joblib` | Bundled trained model |
| `pyproject.toml` | Build config and dependencies |
| `complexity_analyzer.py` | Standalone script (mirrors `_core.py`) |
| `training.py` | Retrain the model |
| `generate_dataset.py` | Generate labeled training data via free LLM APIs |
