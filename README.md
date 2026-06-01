# prompt-complexity-analyzer

ML-powered prompt complexity scorer for LLM routing. Scores any prompt 1–10 and recommends the right model — no API calls, runs locally.

**PyPI package:** `pip install prompt-complexity-analyzer`  
**PyPI page:** https://pypi.org/project/prompt-complexity-analyzer/

---

## Files

| File | Purpose |
|---|---|
| `complexity_analyzer.py` | Core library — import this or run as CLI |
| `training.py` | Train a new ML model from labeled data |
| `generate_dataset.py` | Generate 50k labeled examples via free LLM APIs |
| `my_data.jsonl` | Example labeled dataset |

---

## Quick start

```python
from complexity_analyzer import complexity

r = complexity("prove P≠NP")
print(r.score)   # 9.2
print(r.tier)    # capable
print(r.model)   # claude-opus-4-6
r.explain()      # full breakdown
```

## Retrain with your own data

```bash
# 1. Generate labeled data (free API keys — no credit card)
#    export GROQ_API_KEY=...        console.groq.com
#    export OPENROUTER_API_KEY=...  openrouter.ai
#    export GEMINI_API_KEY=...      aistudio.google.com
python generate_dataset.py --count 10000 --out my_data.jsonl

# 2. Train
python training.py --data my_data.jsonl

# 3. Use your model
python training.py --out model.joblib
```

## CLI

```bash
python complexity_analyzer.py -p "your prompt"
python complexity_analyzer.py --only score -p "your prompt"
python complexity_analyzer.py --provider openai -p "your prompt"
python complexity_analyzer.py --json -p "your prompt"
```
