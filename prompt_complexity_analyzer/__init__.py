"""
prompt_prompt_complexity_analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML-powered prompt complexity analyzer for LLM routing.

Usage:
    from prompt_complexity_analyzer import complexity

    r = complexity("prove P≠NP")
    print(r.score)    # 9.2
    print(r.model)    # claude-opus-4-6
    print(r.tier)     # capable
    r.explain()       # full breakdown

With semantic embeddings (better accuracy):
    from prompt_complexity_analyzer import load_embedding_model
    load_embedding_model()   # call once at startup
    r = complexity("your prompt")
"""

from ._core import (
    complexity,
    extract_features,
    set_model,
    load_embedding_model,
    full_feature_names,
    ComplexityResult,
    FEATURE_NAMES,
    EMBEDDING_DIM,
    MODEL_TIERS,
)

__all__ = [
    "complexity",
    "extract_features",
    "set_model",
    "load_embedding_model",
    "full_feature_names",
    "ComplexityResult",
    "FEATURE_NAMES",
    "EMBEDDING_DIM",
    "MODEL_TIERS",
]

__version__ = "0.1.1"

# ── Auto-load bundled model ───────────────────────────────────────────────────
# model.joblib ships inside the package — no manual set_model() call needed.
from pathlib import Path as _Path

_bundled = _Path(__file__).parent / "model.joblib"
if _bundled.exists():
    set_model(str(_bundled))

# ── Auto-load embedding model ─────────────────────────────────────────────────
# Loads all-MiniLM-L6-v2 automatically if sentence-transformers is installed.
# Install with: pip install "complexity-analyzer[embeddings]"
load_embedding_model()  # silently skips if sentence-transformers not installed
