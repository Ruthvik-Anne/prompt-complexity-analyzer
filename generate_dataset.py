#!/usr/bin/env python3
"""
generate_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates labeled training data using multiple free LLM APIs simultaneously.
Free providers are always tried first; paid only as fallback.

Free capacity per day (no cost, no card):
  Groq         : ~20,000 examples  (Llama 3.3 70B, 1,000 RPD × 20)
  OpenRouter   : ~11,000 examples  (3 free models × 200 RPD × 18)
  Gemini       : ~19,000 examples  (gemini-2.5-flash-lite, 950 RPD × 20)
  ─────────────────────────────────
  Total free   : ~50,000 examples per day → 50k for FREE

Paid fallback (if free quota exhausted, within $10 budget):
  Gemini Flash-Lite: ~$0.30 per 10k examples

Setup (one-time):
    pip install aiohttp
    # Get free API keys (no credit card needed for any of these):
    export GROQ_API_KEY=gsk_...          # console.groq.com
    export OPENROUTER_API_KEY=sk-or-...  # openrouter.ai
    export GEMINI_API_KEY=AIza...        # aistudio.google.com

Usage:
    python generate_dataset.py                     # 50k → dataset.jsonl
    python generate_dataset.py --count 10000
    python generate_dataset.py --free-only         # never charge card
    python generate_dataset.py --resume            # continue existing file
    python generate_dataset.py --out my_data.jsonl

Then retrain:
    python training.py --data dataset.jsonl
"""

from __future__ import annotations

import os, sys, json, re, time, asyncio, argparse, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("\nMissing dependency. Run:\n    pip install aiohttp\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 20   # examples per API request — balances throughput vs token cost

@dataclass
class ModelConfig:
    provider : str
    name     : str            # display name
    model_id : str            # API model string
    base_url : str
    api_key  : str
    rpm      : int            # max requests/minute (set slightly under hard limit)
    rpd      : Optional[int]  # max requests/day  (None = no daily limit)
    free     : bool
    cost_in  : float = 0.0    # $ per 1M input tokens
    cost_out : float = 0.0    # $ per 1M output tokens
    headers  : dict  = field(default_factory=dict)


def build_providers(free_only: bool = False, paid_only: bool = False) -> list[ModelConfig]:
    """Assemble provider list from available env vars. Free models first."""
    configs: list[ModelConfig] = []

    groq_key = os.getenv("GROQ_API_KEY", "")
    or_key   = os.getenv("OPENROUTER_API_KEY", "")
    gem_key  = os.getenv("GEMINI_API_KEY", "")

    # ── Groq  (free, no card) ────────────────────────────────────────────────
    # OpenAI-compatible. Docs: console.groq.com/docs/api-reference
    if groq_key and not paid_only:
        configs.append(ModelConfig(
            provider = "groq",
            name     = "Groq/Llama-3.3-70B",
            model_id = "llama-3.3-70b-versatile",
            base_url = "https://api.groq.com/openai/v1",
            api_key  = groq_key,
            rpm      = 25,    # hard limit 30 — staying under
            rpd      = 950,   # hard limit 1,000 — staying under
            free     = True,
        ))

    # ── OpenRouter free models (no card) ────────────────────────────────────
    # OpenAI-compatible. Docs: openrouter.ai/docs
    # Free models: 20 RPM / 200 RPD each. Models with :free suffix = $0.
    if or_key and not paid_only:
        or_headers = {
            "HTTP-Referer": "https://github.com/complexity-analyzer",
            "X-Title":      "ComplexityDatasetGen",
        }
        for name, mid in [
            ("OR/Llama-3.3-70B",  "meta-llama/llama-3.3-70b-instruct:free"),
            ("OR/DeepSeek-V3",    "deepseek/deepseek-chat-v3-0324:free"),
            ("OR/Qwen3-235B",     "qwen/qwen3-235b-a22b:free"),
        ]:
            configs.append(ModelConfig(
                provider = "openrouter",
                name     = name,
                model_id = mid,
                base_url = "https://openrouter.ai/api/v1",
                api_key  = or_key,
                rpm      = 15,    # hard limit 20 — staying under
                rpd      = 180,   # hard limit 200 — staying under
                free     = True,
                headers  = or_headers,
            ))

    # ── Gemini  (free tier first unless paid_only, then paid) ───────────────
    # Uses Gemini's OpenAI-compatible endpoint.
    # Docs: ai.google.dev/api  |  Auth: Authorization: Bearer KEY
    # NOTE: gemini-2.0-flash and gemini-2.0-flash-lite shutdown June 1 2026 → 404
    # Cheapest current model: gemini-2.5-flash-lite ($0.10/$0.40 per 1M tokens)
    if gem_key:
        gem_base = "https://generativelanguage.googleapis.com/v1beta/openai"

        if not paid_only:
            # Free tier: 15 RPM / 1,000 RPD
            configs.append(ModelConfig(
                provider = "gemini",
                name     = "Gemini-2.5-Flash-Lite (free)",
                model_id = "gemini-2.5-flash-lite",
                base_url = gem_base,
                api_key  = gem_key,
                rpm      = 12,    # hard limit 15 — staying under
                rpd      = 950,   # hard limit 1,000 — staying under
                free     = True,
            ))

        if not free_only:
            # Paid fallback — same model, billing unlocks higher limits
            # $0.10/1M input + $0.40/1M output — ~$1.64 for full 50k
            configs.append(ModelConfig(
                provider = "gemini",
                name     = "Gemini-2.5-Flash-Lite (paid)",
                model_id = "gemini-2.5-flash-lite",
                base_url = gem_base,
                api_key  = gem_key,
                rpm      = 28,
                rpd      = None,   # no daily cap on paid
                free     = False,
                cost_in  = 0.10,   # per 1M tokens
                cost_out = 0.40,
            ))

    # Free models first, then paid
    configs.sort(key=lambda m: (0 if m.free else 1))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Generation Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You generate labeled training data for an AI model-routing system. "
    "Your output must be a valid JSON array only. No markdown, no explanation."
)

def user_prompt(n: int) -> str:
    """Standard diverse generation prompt."""
    return f"""\
Generate exactly {n} diverse prompt-complexity pairs.

COMPLEXITY SCALE (1–10) — cognitive demand on a language model:
1–2  Trivial  : greetings, "2+2", yes/no, single-word answers
2–4  Simple   : basic definitions, easy code, simple Q&A, light creative
4–6  Medium   : multi-step tasks, domain knowledge, structured output
6–8  Complex  : deep analysis, expert domain, multi-part research
8–10 Very Hard: research-level, unsolved/impossible problems, multi-domain synthesis

══ RULE 1 — Complexity = WHAT is asked, NOT length ══
SHORT prompts can be very complex:
  "prove P≠NP"                           → 9.5  (3 words, millennium problem)
  "derive Schrödinger equation"          → 8.5  (3 words, graduate physics)
  "design Byzantine consensus"           → 8.0  (3 words, distributed systems PhD topic)
  "crack RSA-2048"                       → 7.5  (2 words, computationally impossible)

LONG prompts can be trivially simple:
  "Can you please explain in very simple easy-to-understand language what a variable is,
   I am a total beginner who has never coded before" (25 words)   → 2.0  (still just "what is a variable")
  "I would really like to know what 2 plus 2 equals, please give me the answer in a
   simple way that is easy for a beginner" (30 words)             → 1.5  (still just 2+2)

══ RULE 2 — Domain words ≠ complexity ══
These SHORT prompts with domain words are SIMPLE (score 1–3):
  "define array"              → 2.0  (basic CS definition, one sentence answer)
  "what is DNS?"              → 2.0  (basic networking term)
  "translate hello to French" → 1.5  (single word translation)
  "list primary colors"       → 1.5  (trivial knowledge)
  "define variable"           → 2.0  (introductory CS concept)
  "what is a function?"       → 2.0  (basic definition)

These are complex DESPITE having few/no domain words:
  "prove P≠NP"                → 9.5  (unsolved problem)
  "is consciousness physical?" → 5.0  (deep philosophy)
  "model consciousness emergence" → 6.5 (hard problem)

REQUIRED DOMAIN MIX — spread across ALL score ranges:
  casual/chat, math, code, security, medical, legal, finance,
  science, philosophy, history/geopolitics, creative writing,
  everyday life, vague/ambiguous, impossible/open problems

DISTRIBUTION:
  ~15% score 1–3  |  ~20% score 3–5  |  ~30% score 5–7  |  ~25% score 7–9  |  ~10% score 9–10

Return ONLY valid JSON. No markdown fences.
[{{"prompt": "...", "score": 7.5}}, {{"prompt": "...", "score": 2.0}}, ...]"""


def adversarial_prompt(n: int) -> str:
    """Adversarial prompt — specifically generates the hard edge cases.
    Used every 4th batch to ensure the dataset has enough counterexamples."""
    third = n // 3
    return f"""\
Generate exactly {n} prompt-complexity pairs targeting the three hardest cases for ML models.

PART 1 — {third} pairs: SHORT prompts (≤6 words) with HIGH complexity (score 7–10)
Short ≠ simple. These must be genuinely hard despite being brief.
Examples:
  "prove P≠NP"                         → 9.5
  "solve Riemann hypothesis"           → 9.5
  "derive Schrödinger equation"        → 8.5
  "design Byzantine consensus"         → 8.0
  "crack RSA-2048 encryption"          → 7.5
  "prove four color theorem"           → 8.5
  "unify quantum gravity"              → 9.0
Generate {third} NEW examples. Cover: math, CS theory, physics, cryptography, philosophy, distributed systems.

PART 2 — {third} pairs: LONG prompts (≥25 words) with LOW complexity (score 1–4)
Verbose ≠ complex. The underlying task is trivial despite the padding.
Examples:
  "Can you please in very simple terms tell me what hello means, I am a beginner
   who has never heard this word before"                                  → 1.5
  "I would really like to know what 2 plus 2 equals, please explain it in
   the simplest way possible for a complete beginner to math"             → 1.5
Generate {third} NEW long-but-simple examples. Pad trivial tasks (greetings, basic math,
simple definitions) to 25+ words with beginner framing.

PART 3 — {third} pairs: TECHNICAL VOCABULARY but LOW complexity (score 1–3)
Long technical-sounding words ≠ complex. These use domain vocabulary but ask trivial things.
Examples:
  "Define cryptographic fundamentals for complete beginners with zero mathematical background"   → 2.5
  "Explain biochemical nomenclature in simple everyday English for non-scientists"               → 2.0
  "Summarize electromagnetic principles for someone who has never studied physics"               → 2.5
  "What does pharmaceutical terminology mean in simple language for non-medical people?"         → 2.0
  "Describe microprocessor architecture basics for an absolute beginner"                        → 2.5
  "Explain thermodynamic equilibrium to a five year old using simple words"                     → 2.0
  "What does epidemiological surveillance methodology mean in plain English?"                   → 2.0
Generate {third} NEW examples: pick a technical domain word (cryptographic, biochemical,
neurological, electromagnetic, macroeconomic, immunological, etc.) but ask something
a beginner would ask — simple definition, basic explanation, plain-language summary.
Score: 1.5–3.0.

Return ONLY valid JSON. No markdown fences.
[{{"prompt": "...", "score": 9.5}}, {{"prompt": "...", "score": 1.5}}, ...]"""


# Batch counter for alternating prompt types (module-level, shared across workers)
_batch_counter = 0
_batch_lock    = None   # initialized in run()

async def get_prompt() -> str:
    """Returns regular prompt normally, adversarial prompt every 4th batch."""
    global _batch_counter
    async with _batch_lock:
        _batch_counter += 1
        use_adversarial = (_batch_counter % 4 == 0)
    return adversarial_prompt(BATCH_SIZE) if use_adversarial else user_prompt(BATCH_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter — enforces minimum interval between requests."""
    def __init__(self, rpm: int):
        self._interval = 60.0 / max(1, rpm)
        self._last     = 0.0

    async def wait(self):
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            await asyncio.sleep(self._interval - elapsed)
        self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# API Caller
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after

class APIError(Exception):
    pass


async def call_api(session: aiohttp.ClientSession, model: ModelConfig) -> str:
    """Call the OpenAI-compatible chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {model.api_key}",
        "Content-Type":  "application/json",
        **model.headers,
    }
    payload = {
        "model":       model.model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": await get_prompt()},
        ],
        "temperature": 0.95,
        "max_tokens":  3200,
    }

    async with session.post(
        f"{model.base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=90),
    ) as resp:
        if resp.status == 429:
            ra = int(resp.headers.get("retry-after", 60))
            raise RateLimitError(ra)
        if resp.status not in (200, 201):
            body = await resp.text()
            raise APIError(f"HTTP {resp.status}: {body[:300]}")
        data = await resp.json()

    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_examples(raw: str) -> list[dict]:
    """Robustly extract [{prompt, score}] from LLM output."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    for candidate in [cleaned, re.search(r"\[[\s\S]*\]", cleaned)]:
        if not candidate:
            continue
        text = candidate if isinstance(candidate, str) else candidate.group()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return _validate(data)
        except Exception:
            pass
    return []


def _validate(items: list) -> list[dict]:
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        try:
            score = float(item.get("score", -1))
        except (TypeError, ValueError):
            continue
        if len(prompt) >= 3 and 1.0 <= score <= 10.0:
            out.append({"prompt": prompt, "score": round(score, 1)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────

class SharedState:
    def __init__(self, target: int, out_path: str, seen: set[str], existing: int):
        self.target    = target
        self.out_path  = out_path
        self.seen      = seen
        self.total     = existing
        self.by_model : dict[str, int] = {}
        self.cost_usd  = 0.0
        self.start     = time.monotonic()
        self._lock     = asyncio.Lock()
        self._fh       = open(out_path, "a", encoding="utf-8")

    @property
    def done(self) -> bool:
        return self.total >= self.target

    async def write(self, examples: list[dict], model_name: str, cost: float) -> int:
        written = 0
        async with self._lock:
            if self.done:
                return 0
            for ex in examples:
                if self.done:
                    break
                key = hashlib.md5(ex["prompt"].encode()).hexdigest()
                if key in self.seen:
                    continue
                self.seen.add(key)
                self._fh.write(json.dumps({
                    "prompt": ex["prompt"],
                    "score":  ex["score"],
                    "source": model_name,
                }) + "\n")
                written += 1
                self.total += 1
            self._fh.flush()
            self.cost_usd               += cost
            self.by_model[model_name]    = self.by_model.get(model_name, 0) + written
        return written

    def progress_line(self) -> str:
        elapsed = time.monotonic() - self.start
        rate    = self.total / max(elapsed, 1) * 60
        eta_s   = (self.target - self.total) / max(rate / 60, 0.001)
        pct     = min(100.0, self.total / max(self.target, 1) * 100)
        bar     = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        return (f"\r  [{bar}] {self.total:,}/{self.target:,}  "
                f"{rate:.0f}/min  ETA {int(eta_s//60)}m{int(eta_s%60):02d}s  "
                f"Cost ${self.cost_usd:.3f}  ")

    def close(self):
        self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

async def worker(model: ModelConfig, state: SharedState, session: aiohttp.ClientSession):
    limiter          = RateLimiter(model.rpm)
    daily_count      = 0
    consecutive_fail = 0
    MAX_RETRY        = 4
    MAX_CONSEC_FAIL  = 6   # give up on model after this many bad responses in a row

    while not state.done:
        if model.rpd and daily_count >= model.rpd:
            print(f"\n  [{model.name}] daily limit reached ({daily_count} reqs). Done.")
            break

        if consecutive_fail >= MAX_CONSEC_FAIL:
            print(f"\n  [{model.name}] too many consecutive failures — skipping.")
            break

        await limiter.wait()

        success = False
        for attempt in range(MAX_RETRY):
            try:
                raw      = await call_api(session, model)
                examples = parse_examples(raw)

                if not examples:
                    # Bad parse — count as soft failure, try next batch
                    consecutive_fail += 1
                    break

                cost = 0.0
                if not model.free:
                    out_tok = len(raw) // 4
                    cost    = (BATCH_SIZE * 80 / 1e6 * model.cost_in +
                               out_tok    / 1e6 * model.cost_out)

                await state.write(examples, model.name, cost)
                daily_count      += 1
                consecutive_fail  = 0
                success = True
                break

            except RateLimitError as e:
                # Cap sleep at 120s — never block for hours
                wait = min(e.retry_after + 2, 120)
                if attempt < MAX_RETRY - 1:
                    await asyncio.sleep(wait)
                else:
                    # All retries exhausted on rate limit — model is throttled
                    print(f"\n  [{model.name}] rate-limited, cooling down {wait}s...")
                    await asyncio.sleep(wait)
                    consecutive_fail += 1

            except (APIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < MAX_RETRY - 1:
                    await asyncio.sleep(2 ** attempt * 3)
                else:
                    consecutive_fail += 1

        if not success:
            # Brief pause before retrying to avoid tight error loops
            await asyncio.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run(target: int, out_path: str, free_only: bool, paid_only: bool):
    global _batch_lock
    _batch_lock = asyncio.Lock()
    SEP = "━" * 62

    # ── Load existing progress ────────────────────────────────────────────────
    seen: set[str] = set()
    existing = 0
    if Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d   = json.loads(line)
                    key = hashlib.md5(d["prompt"].encode()).hexdigest()
                    seen.add(key)
                    existing += 1
                except Exception:
                    pass
        print(f"  Resuming: found {existing:,} existing examples.")

    if existing >= target:
        print(f"  Already at {existing:,}/{target:,}. Nothing to do.")
        return

    # ── Load providers ────────────────────────────────────────────────────────
    models = build_providers(free_only, paid_only)
    if not models:
        print("\nNo API keys found in environment. Set at least one:\n"
              "  export GROQ_API_KEY=...          # console.groq.com  (free)\n"
              "  export OPENROUTER_API_KEY=...    # openrouter.ai     (free)\n"
              "  export GEMINI_API_KEY=...        # aistudio.google.com\n")
        sys.exit(1)

    free_cap = sum((m.rpd or 0) * BATCH_SIZE for m in models if m.free)
    remaining = target - existing

    print(f"\n{SEP}")
    print(f"  Complexity Dataset Generator")
    print(SEP)
    print(f"  Target      {target:,}   Remaining: {remaining:,}")
    print(f"  Output      {out_path}")
    print(f"  Workers     {len(models)}  (running simultaneously)")
    print(f"  Free cap    ~{free_cap:,} examples today (no cost)")
    if any(not m.free for m in models):
        paid = [m for m in models if not m.free][0]
        est  = remaining / BATCH_SIZE * (BATCH_SIZE * 80 / 1e6 * paid.cost_in
                                        + BATCH_SIZE * 60 / 1e6 * paid.cost_out)
        print(f"  Paid est.   ~${est:.2f} if entirely from paid fallback")
    print()
    print(f"  {'Model':<42} {'Tier':<22} Capacity/day")
    print(f"  {'─'*60}")
    for m in models:
        tier = "FREE" if m.free else f"${m.cost_in:.2f}+${m.cost_out:.2f}/1Mtok"
        cap  = f"{(m.rpd or 0)*BATCH_SIZE:,}" if m.rpd else "unlimited"
        icon = "✓" if m.free else "$"
        print(f"  {icon} {m.name:<41} {tier:<22} {cap}")
    print()
    print(f"  Generating...\n")

    # ── Run workers ───────────────────────────────────────────────────────────
    state = SharedState(target, out_path, seen, existing)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(m, state, session))
            for m in models
        ]

        # Progress display
        while not state.done and not all(t.done() for t in tasks):
            print(state.progress_line(), end="", flush=True)
            await asyncio.sleep(1.5)

        # Final display
        print(state.progress_line(), flush=True)

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    state.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print(f"  Done!")
    print(SEP)
    print(f"  Total      {state.total:,} examples")
    print(f"  Cost       ${state.cost_usd:.4f}")
    print()
    if state.by_model:
        print(f"  Breakdown:")
        for name, cnt in sorted(state.by_model.items(), key=lambda x: -x[1]):
            bar = "█" * min(30, cnt // max(1, state.total // 30))
            print(f"    {name:<42} {cnt:>6,}  {bar}")
    print()
    print(f"  Retrain:")
    print(f"    python training.py --data {out_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate 50k labeled examples using multiple free LLM APIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Free API keys (no credit card required):
  GROQ_API_KEY         →  console.groq.com
  OPENROUTER_API_KEY   →  openrouter.ai
  GEMINI_API_KEY       →  aistudio.google.com  (free tier used first)

Examples:
  python generate_dataset.py
  python generate_dataset.py --count 10000 --free-only
  python generate_dataset.py --paid-only                 # skip free limits, use Gemini paid
  python generate_dataset.py --resume
  python generate_dataset.py --out custom.jsonl
        """,
    )
    parser.add_argument("--count",     type=int, default=50_000, metavar="N",
                        help="Target examples (default: 50000)")
    parser.add_argument("--out",       default="dataset.jsonl",  metavar="PATH",
                        help="Output JSONL file")
    parser.add_argument("--free-only", action="store_true",
                        help="Only use free-tier models — never charge card")
    parser.add_argument("--paid-only", action="store_true",
                        help="Skip free tiers, use paid models only (no rate juggling)")
    parser.add_argument("--resume",    action="store_true",
                        help="Continue from existing output file (auto-detected)")
    args = parser.parse_args()

    if args.free_only and args.paid_only:
        print("Error: --free-only and --paid-only are mutually exclusive.")
        sys.exit(1)

    if not args.resume and Path(args.out).exists():
        ans = input(f"\n  '{args.out}' exists. Resume it? [Y/n]: ").strip().lower()
        if ans not in ("", "y", "yes"):
            Path(args.out).unlink()
            print(f"  Deleted. Starting fresh.\n")

    asyncio.run(run(args.count, args.out, args.free_only, args.paid_only))


if __name__ == "__main__":
    main()