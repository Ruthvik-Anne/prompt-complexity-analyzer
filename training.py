#!/usr/bin/env python3
"""
training.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains an ML model for complexity_analyzer.py.
No GPU, no API, no cloud — runs on any machine with Python.

Requirements:
    pip install scikit-learn joblib numpy

Usage:
    python training.py                     # train and save model.joblib
    python training.py --add               # add your own examples first
    python training.py --out my_model.joblib

After training:
    from complexity_analyzer import complexity, set_model
    set_model("model.joblib")
    r = complexity("your prompt")
    print(r.score, r.backend)   # backend → "ml"
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
except ImportError:
    print("\nMissing dependencies. Run:\n    pip install scikit-learn joblib numpy\n")
    sys.exit(1)

# ── Import feature extractor ──────────────────────────────────────────────────
try:
    from complexity_analyzer import (
        extract_features, FEATURE_NAMES,
        load_embedding_model, full_feature_names,
    )
except ImportError:
    print("\nERROR: complexity_analyzer.py not found.")
    print("Place training.py in the same folder as complexity_analyzer.py.\n")
    sys.exit(1)

# ── Optional: sentence-transformers for semantic embeddings (Option 3) ─────────
_HAS_EMBEDDINGS = False
try:
    import sentence_transformers as _st  # noqa: F401 — just checking availability
    _HAS_EMBEDDINGS = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Labeled Dataset
# Format: (prompt, score)   score: 1 = trivial → 10 = maximally complex
# Wide coverage: trivial, simple, coding, math, creative, medical, legal,
#                finance, security, philosophy, research, multi-part, misc
# ─────────────────────────────────────────────────────────────────────────────

LABELED_DATA: list[tuple[str, float]] = [

    # ── 1.0–1.9  Trivial / Minimal ────────────────────────────────────────────
    ("hi",                                                                     1.0),
    ("ok",                                                                     1.0),
    ("yes",                                                                    1.0),
    ("no",                                                                     1.0),
    ("thanks",                                                                 1.0),
    ("hello there",                                                            1.0),
    ("continue",                                                               1.0),
    ("next",                                                                   1.0),
    ("what is 2+2?",                                                           1.2),
    ("what color is the sky?",                                                 1.2),
    ("is Python a programming language?",                                      1.3),
    ("what does API stand for?",                                               1.4),
    ("what year did World War 2 end?",                                         1.4),
    ("what is the capital of France?",                                         1.4),
    ("how do I say hello in Spanish?",                                         1.5),
    ("translate 'thank you' to French",                                        1.5),
    ("what is the boiling point of water?",                                    1.4),
    ("who wrote Harry Potter?",                                                1.3),
    ("what is 15% of 200?",                                                    1.5),
    ("how many days in a year?",                                               1.2),
    ("is the earth flat?",                                                     1.3),
    ("how tall is Mount Everest?",                                             1.3),
    ("what is 7 * 8?",                                                         1.1),
    ("who is the current Prime Minister of India?",                            1.5),

    # ── 2.0–3.4  Simple ───────────────────────────────────────────────────────
    ("define recursion in programming",                                        2.0),
    ("what is machine learning?",                                              2.2),
    ("list 5 popular programming languages",                                   2.0),
    ("what is the difference between TCP and UDP?",                            2.5),
    ("write a simple Python for loop that prints 1 to 10",                    2.3),
    ("how do I reverse a string in Python?",                                   2.3),
    ("what is a REST API?",                                                    2.5),
    ("explain how HTTP works",                                                 2.5),
    ("what is the time complexity of binary search?",                          2.8),
    ("write a haiku about autumn",                                             2.2),
    ("what should I eat for dinner tonight?",                                  2.0),
    ("write a short joke about programmers",                                   2.0),
    ("what is the meaning of life?",                                           2.8),
    ("recommend a sci-fi book",                                                2.3),
    ("what is a variable in programming?",                                     2.0),
    ("how do I center a div in CSS?",                                          2.2),
    ("explain object-oriented programming simply",                             2.8),
    ("what is the difference between a list and a tuple in Python?",           2.5),
    ("write a hello world program in JavaScript",                              2.0),
    ("how does email work?",                                                   2.5),
    ("what is cloud computing?",                                               2.5),
    ("are you conscious?",                                                     2.8),
    ("help me with stuff",                                                     2.5),
    ("write a bedtime story about a dragon",                                   2.8),
    ("what is blockchain?",                                                    2.8),
    ("how do I fix a 404 error?",                                              2.2),
    ("what is a pointer in C?",                                                2.5),
    ("what is DNS?",                                                           2.3),
    ("write a poem about rain",                                                2.2),
    ("what are some tips to sleep better?",                                    2.2),
    ("what is the difference between RAM and ROM?",                            2.3),
    ("explain what an API key is",                                             2.0),
    ("how do I create a virtual environment in Python?",                       2.2),

    # ── 3.5–4.9  Medium-Low ───────────────────────────────────────────────────
    ("write a Python function to sort a list of dictionaries by a given key",  3.5),
    ("explain how neural networks learn with backpropagation",                 4.0),
    ("compare Python and JavaScript for full-stack web development",           4.0),
    ("write a regex to validate email addresses and explain each part",        3.8),
    ("explain the SOLID principles in OOP with examples",                      4.0),
    ("implement a stack data structure in Python",                             3.5),
    ("explain the differences between SQL and NoSQL databases",                4.0),
    ("how does Git branching work? explain common branching strategies",       4.0),
    ("write a recursive function to compute Fibonacci numbers",                3.5),
    ("explain OAuth 2.0 authentication flow with a diagram",                   4.5),
    ("what are tradeoffs between microservices and monolithic architecture?",  4.5),
    ("write a Python class for a singly linked list with insert and delete",   4.0),
    ("explain how Docker containers work",                                     3.8),
    ("what is the difference between process and thread?",                    3.5),
    ("write a simple REST API using FastAPI with one endpoint",                4.0),
    ("explain the MVC design pattern with a concrete example",                 3.8),
    ("what is memoization? implement it as a Python decorator",               4.0),
    ("write a SQL query to find duplicate rows in a table",                    3.8),
    ("what is gradient descent and how does it work?",                        4.2),
    ("explain supervised vs unsupervised learning with examples",             4.0),
    ("how does the Linux file system hierarchy work?",                        4.0),
    ("implement a queue using two stacks in Python",                          4.0),
    ("explain async/await in JavaScript with examples",                       3.8),
    ("write a Python decorator that logs function calls with timing",          3.8),
    ("what are common SQL injection vulnerabilities and how to prevent them?", 4.5),
    ("explain the OSI model layers with real examples",                       4.0),
    ("write a cover letter for a software engineering role",                  3.5),
    ("should I learn React or Vue for frontend development in 2024?",         3.8),
    ("create a weekly workout plan for a beginner with no equipment",         3.5),
    ("explain the difference between authentication and authorization",       3.5),
    ("how do I use list comprehensions in Python? give examples",             3.5),
    ("what is Big O notation? explain with examples",                         4.0),
    ("write a Python script to read and parse a CSV file",                    3.5),
    ("explain what a hash table is and how it handles collisions",            4.0),

    # ── 5.0–6.4  Medium-High ─────────────────────────────────────────────────
    ("design a URL shortener system with high availability",                   6.0),
    ("analyze time and space complexity of quicksort vs mergesort with code", 5.5),
    ("implement an LRU cache in Python with O(1) get and put",                5.5),
    ("explain how TLS 1.3 handshake works step by step",                      5.5),
    ("compare B-tree, hash, and bitmap database indexing strategies",          6.0),
    ("how would you design a distributed cache like Redis?",                  6.0),
    ("explain CAP theorem and its practical implications",                    6.5),
    ("analyze the Python GIL impact on concurrent code performance",          5.5),
    ("implement a JWT auth system with access and refresh tokens in FastAPI", 5.5),
    ("design a rate limiter for 10k requests per second",                    6.0),
    ("explain how Kubernetes schedules and orchestrates containers",          5.8),
    ("implement a thread-safe producer-consumer queue in Python",             5.5),
    ("write a comprehensive unit test suite for a REST API",                  5.0),
    ("explain CRDT data structures and when to use them over ACID",          6.0),
    ("design a notification system for push, email, and SMS channels",       6.0),
    ("analyze Python memory management: reference counting + GC",            5.5),
    ("write a complete CI/CD pipeline for a Python project with tests",       5.5),
    ("implement WebSocket server for real-time chat",                        5.5),
    ("design a multi-tenant SaaS database schema",                           6.0),
    ("identify and fix race conditions in concurrent Python code",            5.8),
    ("implement Dijkstra's algorithm — explain complexity and edge cases",    5.5),
    ("design an event-driven order management system for e-commerce",        6.0),
    ("explain eBPF and write a simple packet filter",                        6.5),
    ("create a personalised meal plan for a Type 2 diabetic athlete",        5.5),
    ("write a full security review checklist for a web application",         5.5),
    ("explain zero-downtime deployment strategies with rollback plans",      5.5),
    ("teach programming from scratch — full learning roadmap for a beginner",6.0),
    ("analyze the financial viability of launching a SaaS in India",         6.0),
    ("explain differential privacy and implement a Laplace mechanism",       6.0),
    ("design a search autocomplete system that handles typos and ranking",   6.0),
    ("build a web scraper that handles pagination, auth, and rate limits",   5.5),
    ("explain how Postgres handles MVCC and vacuum internally",              5.8),
    ("write a Python async web crawler that respects robots.txt",            5.5),

    # ── 6.5–8.0  Complex ─────────────────────────────────────────────────────
    ("design a zero-knowledge proof system for password auth without revealing the secret", 7.0),
    ("compare RSA, ECDSA, and EdDSA for TLS — recommend for an IoT edge device with justification", 7.5),
    ("design a high-frequency trading system for 1M messages/second with sub-millisecond latency", 7.5),
    ("explain buffer overflow exploitation, write a PoC, and implement mitigations", 7.5),
    ("implement the Raft distributed consensus algorithm from scratch in Python", 8.0),
    ("design a HIPAA-compliant telemedicine platform with E2E encryption and audit trails", 7.5),
    ("analyze geopolitical impact of US-China semiconductor restrictions on Southeast Asian supply chains", 7.0),
    ("build a complete observability stack (metrics + logs + traces) for a microservices system", 7.0),
    ("implement a custom memory allocator in C — first-fit and best-fit — benchmark both", 7.5),
    ("write a Linux kernel module implementing priority-based process scheduling", 8.0),
    ("design a globally distributed database with strong consistency and sub-100ms reads", 8.0),
    ("compare GDPR, India's DPDP Act 2023, and CCPA for a health-tech startup — identify regulatory conflicts", 7.5),
    ("implement a neural network from scratch using only NumPy — then compare with PyTorch on MNIST", 7.5),
    ("design a peer-to-peer encrypted messaging protocol resistant to traffic analysis", 7.5),
    ("perform a STRIDE threat model for a banking mobile application", 7.0),
    ("value a startup using DCF, comparable transactions, and precedent analysis", 7.5),
    ("write an academic literature review on transformer architecture improvements since 2017", 7.0),
    ("design a fraud detection ML system for payments — feature engineering, model selection, monitoring", 7.5),
    ("analyze constitutional implications of AI-generated content across US, EU, and India", 8.0),
    ("implement an adaptive bitrate streaming system for video delivery at scale", 7.5),
    ("design a real-time bidding system for programmatic advertising at 500k QPS", 7.5),
    ("write a comprehensive penetration test report for a financial services API", 7.0),
    ("design a data lakehouse architecture that unifies streaming and batch workloads", 7.5),
    ("model systemic risk propagation in an interbank network using graph theory", 7.5),

    # ── Counterexamples: HIGH WORD LENGTH but SIMPLE ─────────────────────────
    # Teach the model that technical-sounding vocabulary ≠ complexity.
    # These have long avg word lengths but trivially simple underlying tasks.
    ("Define cryptographic fundamentals for complete beginners with zero mathematical background",     2.5),
    ("Explain biochemical nomenclature in simple everyday English for non-scientists",                 2.0),
    ("Summarize electromagnetic principles for someone who has never studied physics before",          2.5),
    ("What does pharmaceutical terminology mean in simple language for non-medical people?",           2.0),
    ("Describe microprocessor architecture basics for an absolute beginner with no technical knowledge",2.5),
    ("Explain thermodynamic equilibrium to a five year old using only simple everyday words",          2.0),
    ("What does epidemiological surveillance methodology mean in plain everyday English?",             2.0),
    ("Explain neurotransmitter chemical mechanisms to someone with absolutely no scientific background",2.5),
    ("Describe photosynthesis biochemistry in the simplest possible language for a primary school student", 2.0),
    ("Can you explain in simple differential algebraic terms what basic addition is for a beginner",   2.0),
    ("What does cardiovascular physiological terminology mean in plain language for non-doctors?",     2.0),
    ("Describe immunological antibody response mechanisms using only simple words a child understands",2.0),
    ("Explain computational algorithmic complexity notation in plain English for someone who cannot code", 2.5),
    ("Summarize macroeconomic recessionary indicators in simple language for someone with no economics background", 2.5),
    ("What does electromagnetic radiation wavelength terminology mean in everyday simple words?",      2.0),

    # ── Counterexamples: SHORT but COMPLEX ───────────────────────────────────
    # Critical — teach the model that brevity ≠ simplicity
    ("prove P≠NP",                                                             9.0),
    ("solve the halting problem",                                              9.0),
    ("prove the Riemann hypothesis",                                           9.0),
    ("derive the Black-Scholes equation from first principles",                7.5),
    ("prove RSA is CCA-secure",                                                8.0),
    ("design a Sybil-resistant consensus mechanism",                           7.5),
    ("formally verify this concurrent algorithm",                              7.5),
    ("analyze RLHF's alignment guarantees",                                    7.0),
    ("evaluate India's shadow banking systemic risk",                          7.0),
    ("prove sqrt(2) is irrational",                                            4.0),
    ("explain Gödel's incompleteness theorems",                                6.5),
    ("is consciousness reducible to physical processes?",                      5.0),
    ("what are the implications of the halting problem for AI?",               6.0),
    ("compare CRYSTALS-Kyber and RSA for post-quantum migration",              7.5),
    ("design a Byzantine fault-tolerant consensus protocol",                   8.0),
    ("model systemic contagion in an interbank network",                       7.5),
    ("prove the four color theorem",                                           8.5),
    ("derive Einstein's field equations",                                      8.0),
    ("analyze Nash equilibrium in a multi-player zero-sum game",               6.5),
    ("what are the security guarantees of Signal protocol?",                   6.0),
    ("critique the GDPR's adequacy for AI systems",                            6.5),
    ("design a zero-knowledge proof for set membership",                       7.5),
    ("is free will compatible with determinism?",                              5.0),

    # ── Counterexamples: LONG but SIMPLE ─────────────────────────────────────
    # Verbose basic questions — teach the model that length ≠ complexity
    ("Can you please explain to me in the simplest possible way, using very basic language that a total beginner with no background in programming at all would understand, what a variable is in programming? Please keep it very basic.", 2.0),
    ("I am just starting to learn Python and I was wondering if you could show me how to write a for loop, just a very simple one that prints numbers from 1 to 10, nothing complicated at all, just the most basic example you can give me.", 2.0),
    ("Hello, I would just like to know, in very simple terms that anyone could understand, what the difference is between the internet and the world wide web. I have heard people use these terms interchangeably but I think they mean different things?", 2.2),
    ("I was wondering if you could help me understand what a function is in programming. I am a complete beginner, I have never coded before, and I just want a very simple explanation with maybe one easy example.", 2.0),
    ("Could you please write me a very short and very simple story about a cat who goes on a small adventure in a garden? It does not need to be long or complicated, just a fun little story that is easy to read.", 2.2),
    ("I am trying to understand what the cloud is. I have heard this word a lot but I am not really sure what it means. Can you explain it in very simple everyday language as if I had no technical background whatsoever?", 2.2),
    ("Please write a very simple Python script that just asks the user for their name and then says hello to them. I am a beginner so please keep it as simple as possible, just the most basic version of this program.", 2.0),
    ("I just want to know a simple way to remember the difference between affect and effect in English. Can you give me a simple trick or rule? I always get confused by these two words.", 1.8),

    # ── 8.0–10.0  Very Complex ────────────────────────────────────────────────
    ("synthesize a post-quantum cryptography migration strategy for a national PKI — evaluate CRYSTALS-Kyber, Dilithium, SPHINCS+, compare against RSA/ECC, model transition risk, propose a 5-year roadmap", 9.5),
    ("design a complete autonomous vehicle perception stack — compare LiDAR-only vs camera-LiDAR fusion, implement Kalman filter sensor fusion, design path planning, and verify safety guarantees formally", 9.5),
    ("analyze macroeconomic factors driving INR/USD over 2020-2024, build a quantitative forecasting model, back-test it, identify failure modes, and propose hedging strategies for a mid-size Indian exporter", 8.5),
    ("compare 10 distributed databases (Cassandra, CockroachDB, TiDB, YugabyteDB, Vitess, Citus, Spanner, DynamoDB, CosmosDB, FaunaDB) across consistency, latency, cost, and operations — recommend for three use cases", 9.0),
    ("design a real-time fraud detection system for UPI at 50k TPS — ML pipeline, feature store, streaming, explainability, regulatory compliance, and rollback", 9.0),
    ("full legal analysis of liability in autonomous vehicle accidents under Indian, US, and EU law — identify gaps, propose frameworks, analyze insurance implications", 8.5),
    ("implement a complete compiler for a C subset — lexer, parser, AST, semantic analysis, IR, x86 codegen — add constant folding and dead code elimination", 9.5),
    ("design a privacy-preserving federated learning system for medical imaging across 50 hospitals — data heterogeneity, Byzantine fault tolerance, differential privacy, and compliance", 9.5),
    ("analyze quantum computing's threat to financial cryptographic infrastructure — timeline to risk, post-quantum migration complexity for SWIFT, systemic financial risk model", 9.0),
    ("create a comprehensive deep-tech startup business plan — market analysis, competitive landscape, tech moat, 5-year DCF, regulatory roadmap, go-to-market, team, and risk", 9.0),
    ("research and synthesize all known approaches to the P vs NP problem — assess each approach's progress, identify promising directions, and estimate probability of resolution in 10 years", 9.5),
    ("design and formally verify a Byzantine fault-tolerant consensus protocol for a mission-critical system — prove safety and liveness, implement, and benchmark at scale", 9.5),

]


# ─────────────────────────────────────────────────────────────────────────────
# Build Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert (prompt, score) pairs → (X, y) arrays using full_feature_names()."""
    names   = full_feature_names()
    X_rows, y_vals = [], []
    for prompt, score in data:
        feats = extract_features(prompt)
        X_rows.append([feats[f] for f in names])
        y_vals.append(score)
    return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(data: list[tuple[str, float]], output_path: str = "model.joblib") -> None:
    SEP = "━" * 58
    print(f"\n{SEP}")
    print("  Complexity Analyzer — ML Training")
    print(SEP)

    # ── Option 3: load embedding model if available ───────────────────────────
    if _HAS_EMBEDDINGS:
        print(f"\n  Loading sentence-transformer (all-MiniLM-L6-v2)...")
        ok = load_embedding_model()
        if ok:
            n_feats = len(full_feature_names())
            print(f"  Semantic embeddings enabled. Total features: {n_feats} (17 keyword + 384 embedding)")
        else:
            n_feats = len(full_feature_names())
            print(f"  Embedding load failed — using {n_feats} keyword features only.")
    else:
        n_feats = len(full_feature_names())
        print(f"\n  sentence-transformers not installed — using {n_feats} keyword features only.")
        print(f"  For better accuracy: pip install sentence-transformers")

    # 1. Build dataset
    print(f"\n[1/5] Building feature matrix from {len(data)} labeled prompts...")
    X, y = build_dataset(data)
    print(f"      Features : {X.shape[1]}  |  Samples : {X.shape[0]}")
    print(f"\n      Score distribution:")
    for lo, hi in [(1, 3), (3, 5), (5, 7), (7, 9), (9, 11)]:
        count = int(((y >= lo) & (y < hi)).sum())
        bar   = "█" * count + "░" * max(0, 20 - count)
        print(f"      {lo}–{hi}  {bar}  {count} examples")

    # 2. Train/test split
    print(f"\n[2/5] Splitting: 80% train / 20% test (stratified by score bins)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 3. Train
    print(f"\n[3/5] Training GradientBoostingRegressor...")
    print(f"      (Fast tree-based model — no GPU needed, works great on 18 features)")
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"      Done.")

    # 4. Evaluate
    print(f"\n[4/5] Evaluating...")
    y_pred = np.clip(model.predict(X_test), 1.0, 10.0)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5

    cv     = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv.mean()

    print(f"\n      Test MAE      : {mae:.3f}  ← avg error in score points")
    print(f"      Test RMSE     : {rmse:.3f}")
    print(f"      5-fold CV MAE : {cv_mae:.3f}")

    if   cv_mae < 0.8:  verdict = "✅ Excellent"
    elif cv_mae < 1.2:  verdict = "✅ Good"
    elif cv_mae < 1.8:  verdict = "⚠  Acceptable — add more labeled data to improve"
    else:               verdict = "❌ Needs more labeled data"
    print(f"      Verdict       : {verdict}")

    # Feature importance
    print(f"\n      Top features the model learned:")
    ranked = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
    for name, imp in ranked[:8]:
        bar = "█" * int(imp * 200)
        print(f"      {name:<32} {bar}  {imp:.3f}")

    # 5. Save
    print(f"\n[5/5] Saving model → {output_path}")
    joblib.dump(model, output_path)
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"      Saved. ({size_kb} KB)")

    print(f"\n{SEP}")
    print("  Done! Load your model:")
    print(SEP)
    print(f"""
    from complexity_analyzer import complexity, set_model
    set_model("{output_path}")

    r = complexity("your prompt here")
    print(r)           # Score 6.2/10 | Tier: balanced | Backend: ml
    print(r.score)     # 6.2
    print(r.model)     # claude-sonnet-4-6
""")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive label addition
# ─────────────────────────────────────────────────────────────────────────────

def add_labels_interactively() -> list[tuple[str, float]]:
    print("\nAdd your own labeled prompts. Type 'done' when finished.")
    print("Guide: 1=trivial  3=simple  5=medium  7=complex  9=very complex\n")
    extra: list[tuple[str, float]] = []
    while True:
        prompt = input("Prompt (or 'done'): ").strip()
        if prompt.lower() == "done":
            break
        if not prompt:
            continue
        try:
            score = float(input("Score (1–10): ").strip())
            score = max(1.0, min(10.0, score))
            extra.append((prompt, score))
            print(f"  ✓ Added  score={score:.1f}\n")
        except ValueError:
            print("  Invalid score, skipped.\n")
    return extra


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[tuple[str, float]]:
    """Load (prompt, score) pairs from a JSONL file (from generate_dataset.py)."""
    out: list[tuple[str, float]] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d      = json.loads(line)
                prompt = str(d.get("prompt", "")).strip()
                score  = float(d.get("score", -1))
                if prompt and 1.0 <= score <= 10.0:
                    out.append((prompt, score))
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    if skipped:
        print(f"      (skipped {skipped} malformed lines)")
    return out


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Train ML model for complexity_analyzer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training.py
  python training.py --data dataset.jsonl          # from generate_dataset.py
  python training.py --data dataset.jsonl --add    # generated + your own
  python training.py --add
  python training.py --out my_model.joblib
        """,
    )
    parser.add_argument("--data", metavar="PATH",
                        help="JSONL file from generate_dataset.py (merged with built-in)")
    parser.add_argument("--add",  action="store_true",
                        help="Interactively add your own labeled examples")
    parser.add_argument("--out",  default="model.joblib", metavar="PATH",
                        help="Output model path (default: model.joblib)")
    args = parser.parse_args()

    data = list(LABELED_DATA)

    if args.data:
        if not __import__("pathlib").Path(args.data).exists():
            print(f"Error: file not found: {args.data}")
            __import__("sys").exit(1)
        generated = load_jsonl(args.data)
        data.extend(generated)
        print(f"\n  Loaded {len(generated):,} examples from {args.data}")
        print(f"  Built-in: {len(LABELED_DATA)}  |  Generated: {len(generated)}  |  Total: {len(data):,}")

    if args.add:
        extra = add_labels_interactively()
        if extra:
            data.extend(extra)
            print(f"\nTotal: {len(data)} examples ({len(extra)} yours)\n")

    train(data, output_path=args.out)