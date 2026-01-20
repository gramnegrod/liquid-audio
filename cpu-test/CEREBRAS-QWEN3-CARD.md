# Cerebras Qwen3-32B Integration Card

**Model**: `qwen-3-32b` via Cerebras Cloud
**Speed**: 2,400 tokens/sec (8x faster than Groq Llama 3.3 70B)
**API**: OpenAI-compatible

---

## Quick Start

```bash
# 1. Get API key from https://cloud.cerebras.ai/

# 2. Add to .env
echo "CEREBRAS_API_KEY=csk-your-key-here" >> .env

# 3. Run
python app_perplexity_stack.py

# 4. Open http://localhost:5008
```

---

## Critical: Thinking Tags

Qwen3 is a "thinking" model that outputs reasoning in `<think>` tags by default. **This will break TTS** if not handled.

### Problem
```
<think>
Okay, the user is asking about the weather. Let me think about this...
</think>

The weather is sunny today.
```

### Solution (Already Implemented)

**1. System prompt instruction:**
```python
SYSTEM_PROMPT = """...
IMPORTANT: Respond directly. Do NOT use <think> tags or show your reasoning.
..."""
```

**2. Regex stripping (backup):**
```python
import re
reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)  # Closed tags
reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL)  # Unclosed tags
reply = reply.strip()
```

Both are needed - the system prompt reduces thinking, the regex catches any that slip through.

---

## API Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

response = client.chat.completions.create(
    model="qwen-3-32b",
    messages=[
        {"role": "system", "content": "Respond directly. Do NOT use <think> tags."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=200,
    temperature=0.7,
)

reply = response.choices[0].message.content
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│ ROUTER (Groq Llama 3.1 8B)  │  Fast search detection
│ "Does this need web search?"│  ~50ms
└─────────────────────────────┘
    │                │
    │ NO             │ YES
    ▼                ▼
┌─────────────┐  ┌─────────────────────┐
│ Cerebras    │  │ Perplexity Search   │
│ Qwen3-32B   │  │ + Cerebras reformat │
│ 2,400 t/s   │  │                     │
└─────────────┘  └─────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Cartesia TTS (Sonic 2)      │
└─────────────────────────────┘
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CEREBRAS_API_KEY` | Yes | Get from https://cloud.cerebras.ai/ |
| `GROQ_API_KEY` | Yes | For ASR (Whisper) and Router |
| `PERPLEXITY_API_KEY` | Yes | For web search |
| `CARTESIA_API_KEY` | Yes | For TTS |

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/chat` | POST | SSE streaming (browser) |
| `/sts-sync` | POST | Sync JSON (test harness) |
| `/health` | GET | Status check |

---

## Test Harness

```bash
# Run test
python -m sts_harness.agents.sts_test_agent http://localhost:5008/sts-sync greeting

# Expected output:
# ✓ Verdict: YES
# Pass rate: 100%
# TTFA: ~300ms
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `<think>` tags in response | Model outputting reasoning | Add system prompt instruction + regex strip |
| Empty response after strip | Model only output thinking | Increase `max_tokens` (200 minimum) |
| TTS 400 error | Empty text sent to Cartesia | Check response isn't empty before TTS |
| Slow responses | Model thinking too much | System prompt: "Be concise, respond directly" |

---

## Files Modified

- `cpu-test/app_perplexity_stack.py` - Main application
- `cpu-test/.env` - API keys
- `cpu-test/PERPLEXITY-VOICE-ASSISTANT.md` - Full documentation

---

## Costs

Cerebras pricing (as of Jan 2026): ~$0.10 per 1M tokens
Much cheaper than OpenAI, comparable to Groq.

---

*Last updated: January 19, 2026*
