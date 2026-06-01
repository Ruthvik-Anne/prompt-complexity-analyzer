#!/usr/bin/env python3
"""
prompt_complexity_analyzer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-file prompt complexity analyzer.
ML-powered when a trained model is available; heuristic fallback.

── Import API ────────────────────────────────────────────────────
    from prompt_complexity_analyzer import complexity

    r = complexity("Your prompt here")
    r.score          # float 1–10
    r.tier           # "fast" | "balanced" | "capable"
    r.model          # recommended model string
    r["reasoning"]   # dimension score — fuzzy key access
    print(r)         # one-line summary
    r.explain()      # full breakdown string

── With ML model ─────────────────────────────────────────────────
    from prompt_complexity_analyzer import set_model
    set_model("./model.joblib")       # load once globally

    # or per-call:
    r = complexity("prompt", model_path="./model.joblib")

── Feature extraction (for training) ────────────────────────────
    from prompt_complexity_analyzer import extract_features, FEATURE_NAMES

    feats = extract_features("Your prompt")      # dict[str, float]
    X     = [feats[f] for f in FEATURE_NAMES]   # list in canonical order
    # Train: regressor X → score (float 1–10)
    # Save:  import joblib; joblib.dump(model, "model.joblib")

── CLI ───────────────────────────────────────────────────────────
    python prompt_complexity_analyzer.py -p "Your prompt"
    python prompt_complexity_analyzer.py --only score -p "..."
    python prompt_complexity_analyzer.py --only reasoning -p "..."
    python prompt_complexity_analyzer.py --provider openai -p "..."
    python prompt_complexity_analyzer.py --model ./model.joblib -p "..."
    python prompt_complexity_analyzer.py --json -p "..."
"""

from __future__ import annotations

import re
import sys
import json
import argparse
from typing import Any, Optional

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

# ─────────────────────────────────────────────────────────────────────────────
# Keyword Banks
# ─────────────────────────────────────────────────────────────────────────────

_REASONING_KW = [
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "critique", "synthesize", "synthesise", "step by step", "step-by-step",
    "think through", "pros and cons", "trade-offs", "tradeoffs",
    "implications", "consequences", "justify", "argue", "prove", "disprove",
    "infer", "deduce", "derive", "refute", "explain why", "reason through",
    "break down", "root cause", "first principles", "critically",
]

_DOMAIN_KW: dict[str, list[str]] = {
    "math":     ["integral", "derivative", "calculus", "theorem", "proof",
                 "polynomial", "matrix", "vector", "eigenvalue", "probability",
                 "statistics", "combinatorics", "modular arithmetic", "fourier"],
    "code":     ["function", "class", "algorithm", "debug", "refactor",
                 "implement", "compile", "runtime", "async", "api",
                 "database", "optimize", "sql", "regex", "recursion",
                 "binary", "complexity", "data structure"],
    "security": ["exploit", "vulnerability", "penetration", "ctf", "cve",
                 "payload", "injection", "xss", "csrf", "rop", "shellcode",
                 "privilege escalation", "reverse shell", "buffer overflow",
                 "ecdh", "rsa", "tls", "cipher", "cryptograph"],
    "medical":  ["diagnosis", "symptom", "treatment", "pharmacology",
                 "clinical", "pathology", "prognosis", "dosage",
                 "contraindication", "etiology", "differential"],
    "legal":    ["liability", "statute", "jurisdiction", "precedent",
                 "contract", "intellectual property", "tort", "litigation",
                 "compliance", "gdpr", "dpdp", "regulatory"],
    "finance":  ["portfolio", "derivative", "arbitrage", "hedge",
                 "valuation", "amortization", "liquidity", "sharpe ratio",
                 "volatility", "dcf", "ebitda", "options pricing", "equity"],
    "science":  ["hypothesis", "empirical", "thermodynamics", "quantum",
                 "molecular", "genome", "entropy", "catalysis", "osmosis",
                 "photosynthesis", "relativity", "atomic"],
}

_AMBIGUITY_KW = [
    "something", "stuff", "things", "whatever", "anything", "somehow",
    "maybe", "perhaps", "not sure", "i think", "kind of", "sort of",
    "some kind", "you know", "etc", "and so on", "and stuff",
]

_CREATIVE_KW = [
    "write", "compose", "generate", "draft", "brainstorm", "imagine",
    "invent", "story", "poem", "essay", "narrative", "fiction",
    "come up with", "create a",
]

_OUTPUT_FORMAT_KW: dict[str, list[str]] = {
    "json_yaml_xml": ["json", "yaml", "xml", "schema", "format as", "structured output"],
    "long_form":     ["essay", "report", "article", "in-depth", "comprehensive", "detailed"],
    "code_output":   ["```", "implement", "code", "script", "program"],
    "table":         ["table", "spreadsheet", "compare side by side", "columns"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature Names  (canonical order — do not reorder; matches ML model input)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Structural (no token_count — verbosity ≠ complexity)
    "sentence_count",
    "avg_word_length",            # suppressed: binned 0/1/2 to limit dominance
    "question_count",
    "subtask_signal_count",       # explicit multi-part signals
    # Reasoning demand
    "reasoning_kw_count",
    # Domain expertise required
    "domain_math_count",
    "domain_code_count",
    "domain_security_count",
    "domain_medical_count",
    "domain_legal_count",
    "domain_finance_count",
    "domain_science_count",
    "unique_domain_count",        # cross-domain breadth
    # Output and intent
    "ambiguity_kw_count",
    "creative_kw_count",
    "output_format_signal_count",
    "has_code_block",
]  # 17 keyword features

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ─────────────────────────────────────────────────────────────────────────────
# Model Tiers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_TIERS: dict[str, dict[str, Any]] = {
    "fast": {
        "label":       "Fast / Lightweight",
        "anthropic":   "claude-haiku-4-5",
        "openai":      "gpt-4o-mini",
        "google":      "gemini-2.0-flash",
        "ollama":      "qwen3:1.7b",
        "score_range": (1.0, 3.5),
        "use_when":    "Factual Q&A, simple rewrites, classification, lookups",
    },
    "balanced": {
        "label":       "Balanced",
        "anthropic":   "claude-sonnet-4-6",
        "openai":      "gpt-4o",
        "google":      "gemini-2.0-pro",
        "ollama":      "qwen3:14b",
        "score_range": (3.6, 6.5),
        "use_when":    "Multi-step reasoning, code, domain tasks, structured output",
    },
    "capable": {
        "label":       "High Capability",
        "anthropic":   "claude-opus-4-6",
        "openai":      "o1",
        "google":      "gemini-2.5-pro",
        "ollama":      "qwen3:32b",
        "score_range": (6.6, 10.0),
        "use_when":    "Complex research, deep reasoning, ambiguous high-stakes tasks",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ML Model State
# ─────────────────────────────────────────────────────────────────────────────

_ml_model: Optional[Any] = None


def set_model(path: str) -> None:
    """
    Load a trained ML model globally. Call once at startup.

    The model must implement sklearn's predict(X) interface where
    X is shape (n_samples, len(full_feature_names())) and output is a
    float score in [1, 10].

    Compatible save/load:
        import joblib
        joblib.dump(trained_model, "model.joblib")   # save
        set_model("model.joblib")                     # load here
    """
    global _ml_model
    _ml_model = _load_model_from_path(path)


def _load_model_from_path(path: str) -> Any:
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Embedding Model State ─────────────────────────────────────────────────────

_embedding_model: Optional[Any] = None


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> bool:
    """
    Load a sentence-transformer for semantic embeddings (Option 3).
    Downloads ~80MB once on first call, then cached locally.

    Returns True if loaded successfully, False otherwise.

        pip install sentence-transformers
        from prompt_complexity_analyzer import load_embedding_model
        load_embedding_model()       # downloads all-MiniLM-L6-v2 once
        r = complexity("your prompt")
    """
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _embedding_model = SentenceTransformer(model_name)
        return True
    except ImportError:
        print(
            "[prompt_complexity_analyzer] sentence-transformers not installed. "
            "Run: pip install sentence-transformers",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(
            f"[prompt_complexity_analyzer] Could not load embedding model ({e}). "
            "Continuing without embeddings — keyword features only.",
            file=sys.stderr,
        )
        return False


def full_feature_names() -> list[str]:
    """
    Returns the complete feature vector names used by the ML model.
    = FEATURE_NAMES (16 keyword features)
    + emb_000..emb_383 (384 semantic embedding dims, if embedding model loaded)

    Use this in training.py to build the feature matrix:
        X = [feats[f] for f in full_feature_names()]
    """
    if _embedding_model is not None:
        return FEATURE_NAMES + [f"emb_{i:03d}" for i in range(EMBEDDING_DIM)]
    return FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(prompt: str) -> dict[str, float]:
    """
    Extract features from a prompt.
    - 16 keyword features (FEATURE_NAMES)
    - + 384 semantic embedding dims if load_embedding_model() was called

    Returns a dict keyed by full_feature_names().
    Use to generate training data:

        label = float(input("Score (1-10): "))
        row   = {"features": extract_features(prompt), "label": label}

    Feature vector for model:
        X = [feats[f] for f in full_feature_names()]
    """
    tl    = prompt.lower()
    words = re.findall(r"\b\w+\b", tl)
    sents = [s for s in re.split(r"[.!?]+", prompt) if s.strip()]

    # ── Structural ────────────────────────────────────────────────────────────
    sentence_count  = float(max(1, len(sents)))
    question_count  = float(len(re.findall(r"\?", prompt)))

    # avg_word_length binned to 0/1/2 — preserves directional signal
    # (simple/medium/technical) without letting the continuous value dominate.
    # 0 = short words (≤4.5 avg)  1 = medium (4.5–6.0)  2 = long/technical (>6.0)
    _awl = (sum(len(w) for w in words) / len(words)) if words else 0.0
    avg_word_length = 0.0 if _awl < 4.5 else (1.0 if _awl < 6.0 else 2.0)

    _subtask_patterns = [
        r"\balso\b", r"\badditionally\b", r"\bfurthermore\b",
        r"\band then\b", r"\bmoreover\b", r"\bfinally\b",
        r"\bstep \d+", r"^\s*\d+[.)]\s", r"\?(?=\s|$)",
    ]
    subtask_signal_count = float(sum(
        len(re.findall(p, tl, re.MULTILINE)) for p in _subtask_patterns
    ))

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasoning_kw_count = float(sum(1 for kw in _REASONING_KW if kw in tl))

    # ── Domain ────────────────────────────────────────────────────────────────
    domain_counts: dict[str, int] = {
        d: sum(1 for kw in kws if kw in tl)
        for d, kws in _DOMAIN_KW.items()
    }
    unique_domain_count = float(sum(1 for c in domain_counts.values() if c > 0))

    # ── Other signals ─────────────────────────────────────────────────────────
    ambiguity_kw_count         = float(sum(1 for kw in _AMBIGUITY_KW if kw in tl))
    creative_kw_count          = float(sum(1 for kw in _CREATIVE_KW   if kw in tl))
    output_format_signal_count = float(sum(
        1 for kws in _OUTPUT_FORMAT_KW.values() if any(kw in tl for kw in kws)
    ))
    has_code_block = 1.0 if "```" in prompt else 0.0

    feats: dict[str, float] = {
        "sentence_count":            sentence_count,
        "avg_word_length":           avg_word_length,
        "question_count":            question_count,
        "subtask_signal_count":      subtask_signal_count,
        "reasoning_kw_count":        reasoning_kw_count,
        "domain_math_count":         float(domain_counts["math"]),
        "domain_code_count":         float(domain_counts["code"]),
        "domain_security_count":     float(domain_counts["security"]),
        "domain_medical_count":      float(domain_counts["medical"]),
        "domain_legal_count":        float(domain_counts["legal"]),
        "domain_finance_count":      float(domain_counts["finance"]),
        "domain_science_count":      float(domain_counts["science"]),
        "unique_domain_count":       unique_domain_count,
        "ambiguity_kw_count":        ambiguity_kw_count,
        "creative_kw_count":         creative_kw_count,
        "output_format_signal_count":output_format_signal_count,
        "has_code_block":            has_code_block,
    }

    # ── Semantic embeddings (Option 3) ────────────────────────────────────────
    if _embedding_model is not None:
        embedding = _embedding_model.encode(prompt, show_progress_bar=False)
        for i, val in enumerate(embedding):
            feats[f"emb_{i:03d}"] = float(val)

    return feats


def _feature_vector(prompt: str) -> list[float]:
    feats = extract_features(prompt)
    return [feats[f] for f in full_feature_names()]


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Scoring  (fallback when no ML model is available)
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_score(feats: dict[str, float]) -> tuple[float, dict[str, float]]:
    rk = feats["reasoning_kw_count"]
    sk = feats["subtask_signal_count"]
    ak = feats["ambiguity_kw_count"]
    of = feats["output_format_signal_count"]
    ud = feats["unique_domain_count"]
    wl = feats["avg_word_length"]   # binned: 0/1/2
    total_domain = sum(feats[f"domain_{d}_count"]
                       for d in ["math", "code", "security", "medical", "legal", "finance", "science"])

    dims: dict[str, float] = {}

    # Vocabulary (binned 0/1/2 → weak directional signal only)
    dims["Vocabulary"] = {0.0: 2.0, 1.0: 4.5, 2.0: 6.5}.get(wl, 2.0)

    # Multi-part
    if   sk == 0: dims["Multi-part"] = 1.5
    elif sk <= 2: dims["Multi-part"] = 4.5
    elif sk <= 5: dims["Multi-part"] = 7.0
    else:         dims["Multi-part"] = 9.5

    # Reasoning demand
    dims["Reasoning Depth"] = min(10.0, 1.5 + rk * 1.8) if rk > 0 else 1.5

    # Domain specificity
    if total_domain == 0:
        dims["Domain Specificity"] = 2.0
    else:
        dims["Domain Specificity"] = min(10.0, 3.0 + total_domain * 1.2 + (ud - 1) * 1.5)

    # Ambiguity
    if   ak >= 3: dims["Ambiguity"] = 8.0
    elif ak >= 1: dims["Ambiguity"] = 5.0
    else:         dims["Ambiguity"] = 2.0

    # Output complexity
    dims["Output Complexity"] = min(10.0, of * 3.0) if of > 0 else 1.5

    weights = {
        "Vocabulary":         0.05,   # suppressed — binned, weak signal
        "Multi-part":         0.22,
        "Reasoning Depth":    0.35,
        "Domain Specificity": 0.25,
        "Ambiguity":          0.08,
        "Output Complexity":  0.05,
    }
    total_w = sum(weights.values())
    overall = sum(dims[d] * weights[d] for d in dims) / total_w
    return round(min(10.0, max(1.0, overall)), 1), dims


# ─────────────────────────────────────────────────────────────────────────────
# ComplexityResult
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityResult:
    """
    Full result of a complexity analysis.

    Attributes:
        score       Overall complexity score (float, 1–10).
        tier        "fast" | "balanced" | "capable"
        model       Recommended model string for the chosen provider.
        provider    Provider used (anthropic / openai / google / ollama).
        label       Human-readable tier label.
        dimensions  {dimension_name: score} dict.
        flags       List of advisory messages.
        backend     "ml" or "heuristic".
        features    Raw extracted features ({FEATURE_NAMES key: value}).

    Key access (fuzzy substring match on dimension names + special fields):
        result["reasoning"]  →  Reasoning Depth score
        result["score"]      →  overall score float
        result["tier"]       →  tier string
        result["model"]      →  recommended model
        result["backend"]    →  "ml" or "heuristic"

    Methods:
        result.get(key, default)  →  same as [] but returns default on miss
        result.explain()          →  full formatted breakdown string
        result.to_dict()          →  dict representation
        result.to_json(indent=2)  →  JSON string
        str(result)               →  one-line summary
    """

    def __init__(
        self,
        score:      float,
        tier:       str,
        provider:   str,
        dimensions: dict[str, float],
        flags:      list[str],
        backend:    str,
        features:   dict[str, float],
    ) -> None:
        self.score      = score
        self.tier       = tier
        self.provider   = provider
        self.dimensions = dimensions
        self.flags      = flags
        self.backend    = backend
        self.features   = features

        _t         = MODEL_TIERS[tier]
        self.model = _t.get(provider, _t["anthropic"])
        self.label = _t["label"]

    # ── Key access ────────────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            valid = "score, tier, model, label, backend, " + ", ".join(self.dimensions)
            raise KeyError(f"'{key}' not found. Valid keys: {valid}")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Fuzzy field access. Substring-matches against dimension names
        and special fields (score, tier, model, label, backend).
        """
        k = key.strip().lower()
        if k in ("score", "overall"):  return self.score
        if k == "tier":                return self.tier
        if k == "model":               return self.model
        if k == "label":               return self.label
        if k == "backend":             return self.backend

        matches = [(n, v) for n, v in self.dimensions.items() if k in n.lower()]
        if not matches:
            return default
        if len(matches) == 1:
            return matches[0][1]
        exact = [(n, v) for n, v in matches if n.lower().startswith(k)]
        return (exact or matches)[0][1]

    # ── String representations ────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"Score {self.score:.1f}/10 | "
            f"Tier: {self.tier} | "
            f"Model: {self.model} | "
            f"Backend: {self.backend}"
        )

    def __repr__(self) -> str:
        return (
            f"ComplexityResult(score={self.score}, tier='{self.tier}', "
            f"model='{self.model}', backend='{self.backend}')"
        )

    def explain(self) -> str:
        """Return a full human-readable breakdown."""
        sep      = "─" * 60
        tier_cfg = MODEL_TIERS[self.tier]

        lines = [
            "",
            f"  {sep}",
            f"  Score      {self.score:>5.1f} / 10",
            f"  Tier       {self.label}",
            f"  Model      {self.model}  [{self.provider}]",
            f"  Backend    {self.backend}",
            f"  Use when   {tier_cfg['use_when']}",
            f"  {sep}",
            "",
            f"  {'Dimension':<24} {'':14}  Score",
            f"  {'─'*58}",
        ]
        for name, score in self.dimensions.items():
            filled = round(score / 10 * 14)
            bar    = "█" * filled + "░" * (14 - filled)
            lines.append(f"  {name:<24} {bar}  {score:>4.1f}")

        if self.flags:
            lines += ["", f"  {'─'*58}"]
            for flag in self.flags:
                lines.append(f"  {flag}")

        lines += [f"  {sep}", ""]
        return "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "score":      self.score,
            "tier":       self.tier,
            "model":      self.model,
            "provider":   self.provider,
            "label":      self.label,
            "backend":    self.backend,
            "dimensions": self.dimensions,
            "flags":      self.flags,
            "features":   self.features,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def complexity(
    prompt:     str,
    provider:   str           = "anthropic",
    model_path: Optional[str] = None,
) -> ComplexityResult:
    """
    Analyze a prompt and return a ComplexityResult.

    Args:
        prompt:     The prompt or task description to analyze.
        provider:   Model provider for the recommendation.
                    One of: "anthropic" (default), "openai", "google", "ollama".
        model_path: Path to a trained ML model (.joblib or .pkl).
                    Overrides the globally loaded model from set_model().
                    Falls back to heuristic if neither is set.

    Returns:
        ComplexityResult

    Examples:
        r = complexity("Explain quantum entanglement simply")
        print(r)                  # one-line summary
        r.explain()               # full breakdown
        r.score                   # 4.2
        r["reasoning"]            # Reasoning Depth score
        r.to_dict()               # serialize

        r = complexity("prompt", provider="openai", model_path="./model.joblib")
    """
    _VALID_PROVIDERS = {"anthropic", "openai", "google", "ollama"}
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(_VALID_PROVIDERS)}")

    feats = extract_features(prompt)
    fvec  = [feats[f] for f in full_feature_names()]

    # Resolve ML model (per-call path takes priority over global)
    model = None
    if model_path:
        model = _load_model_from_path(model_path)
    elif _ml_model is not None:
        model = _ml_model

    # Score
    if model is not None:
        try:
            raw     = float(model.predict([fvec])[0])
            score   = round(min(10.0, max(1.0, raw)), 1)
            _, dims = _heuristic_score(feats)   # dimensions for display
            backend = "ml"
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("feature", "shape", "mismatch", "dimension")):
                print(
                    "[prompt_complexity_analyzer] Feature mismatch — model was trained with embeddings "
                    "but load_embedding_model() was not called (or vice versa). "
                    "Call load_embedding_model() before complexity(). Falling back to heuristic.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[prompt_complexity_analyzer] ML prediction failed ({e}). "
                    "Falling back to heuristic.", file=sys.stderr,
                )
            score, dims = _heuristic_score(feats)
            backend     = "heuristic"
    else:
        score, dims = _heuristic_score(feats)
        backend     = "heuristic"

    # Tier
    if   score <= 3.5: tier = "fast"
    elif score <= 6.5: tier = "balanced"
    else:              tier = "capable"

    # Flags
    flags: list[str] = []
    if score >= 8.5:
        flags.append("⚠  Very high complexity — consider decomposing the task first")
    if dims.get("Ambiguity", 0) >= 7.0:
        flags.append("⚠  High ambiguity — add explicit context / constraints")
    if dims.get("Multi-part", 0) >= 7.0:
        flags.append("💡 Multi-part — consider splitting into sequential sub-prompts")
    if dims.get("Domain Specificity", 0) >= 7.0 and tier == "fast":
        flags.append("💡 Specialized domain at fast tier — may miss nuance")

    return ComplexityResult(
        score=score, tier=tier, provider=provider,
        dimensions=dims, flags=flags, backend=backend, features=feats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="prompt_complexity_analyzer",
        description="Analyze prompt complexity → model routing recommendation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
--only FIELD values:
  score / overall   Overall score (float)
  tier              "fast" | "balanced" | "capable"
  model             Recommended model string
  label             Tier label
  backend           "ml" | "heuristic"
  reasoning         Reasoning Depth score
  domain            Domain Specificity score
  ambiguity         Ambiguity score
  multi             Multi-part score
  length            Length score
  output            Output Complexity score

Examples:
  python prompt_complexity_analyzer.py -p "What is 2+2?"
  python prompt_complexity_analyzer.py -p "Compare ECDH vs RSA for TLS 1.3"
  echo "Some task" | python prompt_complexity_analyzer.py
  python prompt_complexity_analyzer.py --provider openai -p "..."
  python prompt_complexity_analyzer.py --only score -p "..."
  python prompt_complexity_analyzer.py --only reasoning -p "..."
  python prompt_complexity_analyzer.py --model ./model.joblib --only tier -p "..."
  python prompt_complexity_analyzer.py --json -p "..."
""",
    )
    parser.add_argument("-p", "--prompt",
                        help="Prompt text (or pass via stdin)")
    parser.add_argument("--provider",
                        choices=["anthropic", "openai", "google", "ollama"],
                        default="anthropic",
                        help="Model provider (default: anthropic)")
    parser.add_argument("--model", metavar="PATH",
                        help="Path to trained ML model (.joblib or .pkl)")
    parser.add_argument("--only", metavar="FIELD",
                        help="Output a single field value only")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        print("Enter prompt (Ctrl+D when done):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("Error: no prompt provided.", file=sys.stderr)
        sys.exit(1)

    result = complexity(prompt, provider=args.provider, model_path=args.model)

    if args.only:
        value = result.get(args.only)
        if value is None:
            print(
                f"Error: unknown field '{args.only}'. "
                "Try: score, tier, model, label, reasoning, domain, "
                "ambiguity, multi, length, output",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.json:
            print(json.dumps({"field": args.only, "value": value}))
        else:
            print(f"{value:.1f}" if isinstance(value, float) else value)
    elif args.json:
        print(result.to_json())
    else:
        print(result.explain())


if __name__ == "__main__":
    _cli()