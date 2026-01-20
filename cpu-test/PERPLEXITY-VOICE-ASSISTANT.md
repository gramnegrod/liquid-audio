# Perplexity Voice Assistant Stack

A voice assistant using Perplexity's Sonar model for search + synthesis. Perplexity handles grounding internally, producing more accurate responses with less hallucination than raw RAG approaches.

**Target:** ~500ms without search, ~4-6s with search
**Port:** 5008
**Key Advantage:** Perplexity does search AND synthesis - responses are pre-grounded

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER VOICE INPUT                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ASR: Groq Whisper Large v3 Turbo (~200ms)                      │
│  - Fastest Whisper available                                     │
│  - 16kHz mono input                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SEARCH TRIGGER DETECTION                                        │
│  - Keyword matching: "who is", "score", "latest", "stats"...    │
│  - If triggered → Perplexity path                                │
│  - If not → Direct LLM path                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│  DIRECT LLM PATH         │    │  PERPLEXITY PATH (~3-5s)         │
│  - No search needed      │    │                                  │
│  - temp=0.7 (creative)   │    │  Perplexity Sonar does:          │
│  - General conversation  │    │  1. Web search                   │
└──────────────────────────┘    │  2. Result synthesis             │
                                │  3. Citation generation          │
                                │  4. Fact verification            │
                                │                                  │
                                │  Returns pre-grounded answer     │
                                │  with [1], [2] citations         │
                                └──────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM: Cerebras Qwen3-32B @ 2,400 t/s                            │
│  - 8x faster than Groq Llama 3.3 70B                             │
│  - Reformats Perplexity answer for voice                         │
│  - Removes citation brackets [1], [2]                            │
│  - Condenses to 2-3 spoken sentences                             │
│  - Does NOT add any new information                              │
│  (Falls back to Groq if CEREBRAS_API_KEY not set)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TTS: Cartesia Sonic 2 (streaming, ~75ms TTFB)                  │
│  - SSE streaming for low latency                                 │
│  - Gapless pre-scheduled Web Audio playback                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        VOICE OUTPUT                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Perplexity is Better for Grounding

| Approach | How It Works | Hallucination Risk |
|----------|--------------|-------------------|
| **Raw RAG (Tavily)** | Search → Raw results → LLM interprets | Higher - LLM may fill gaps |
| **Perplexity** | Search → Perplexity synthesizes → LLM formats | Lower - answer pre-verified |

**Perplexity's internal process:**
1. Searches multiple sources
2. Cross-references facts
3. Synthesizes a verified answer
4. Includes citations
5. Returns grounded response

Your LLM just reformats for voice - it doesn't interpret or add information.

---

## Tech Stack

| Component | Service | Model/API | Speed |
|-----------|---------|-----------|-------|
| ASR | Groq | `whisper-large-v3-turbo` | ~200ms |
| Search+Synthesis | Perplexity | `sonar` | ~3-5s |
| Router | Groq | `llama-3.1-8b-instant` | ~50ms |
| **Main LLM** | **Cerebras** | **`qwen-3-32b`** | **2,400 t/s** |
| TTS | Cartesia | `sonic-2` | ~75ms TTFB |

> **Note:** If `CEREBRAS_API_KEY` is not set, falls back to Groq `llama-3.3-70b-versatile` @ 280 t/s

---

## Key Code Components

### 1. Perplexity Client Setup

```python
from openai import OpenAI

# Perplexity uses OpenAI-compatible API
perplexity_client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
)
```

### 2. Perplexity Search Function

```python
def web_search(query: str) -> tuple[str, int]:
    """
    Search using Perplexity's sonar model.
    Perplexity does search + synthesis in one call.
    """
    start = time.time()

    response = perplexity_client.chat.completions.create(
        model="sonar",  # or "sonar-pro" for deeper research
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Provide factual, "
                           "well-sourced answers. Include specific data, "
                           "numbers, and cite your sources."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=1000,
        temperature=0.2,  # Low for factual accuracy
    )

    content = response.choices[0].message.content
    latency_ms = int((time.time() - start) * 1000)

    # Build context with citations
    context_parts = [f"PERPLEXITY ANSWER:\n{content}"]

    if hasattr(response, 'citations') and response.citations:
        context_parts.append("\nSOURCES:")
        for i, citation in enumerate(response.citations, 1):
            context_parts.append(f"[{i}] {citation}")

    return "\n".join(context_parts), latency_ms
```

### 3. Conversational System Prompt (SOTA Persona Design)

The conversational prompt uses **SOTA persona consistency techniques** to prevent character drift:

```python
SYSTEM_PROMPT = """You are a voice assistant having a real-time spoken conversation.

IDENTITY:
- You ARE speaking with the user - they hear your voice, you hear theirs
- This is a live audio conversation, not text chat

BEHAVIORAL RULES:
- NEVER say you are "text-based" or cannot hear
- NEVER say you don't have access to conversation history when you do
- Use the conversation history in messages to answer recall questions accurately

RESPONSE STYLE:
- 2-3 sentences unless more detail requested
- Conversational, natural spoken rhythm
- Direct answers without hedging"""
```

**Why this structure works (based on [LearnPrompting](https://learnprompting.org/docs/advanced/zero_shot/role_prompting) research):**
- **IDENTITY layer** - Direct "You ARE" statements, not "Imagine you are"
- **BEHAVIORAL RULES** - Explicit "NEVER do X" constraints prevent character breaking
- **RESPONSE STYLE** - Separates formatting from identity

### 4. Voice Formatting System Prompt (Search Results)

Since Perplexity already grounded the response, the LLM just formats for voice:

```python
SYSTEM_PROMPT_WITH_SEARCH = """You are reformatting a Perplexity search result for voice output.

The user's message contains a PERPLEXITY ANSWER that has already been researched and verified.

Your job:
1. Condense the Perplexity answer into 2-3 natural spoken sentences
2. Keep the key facts and numbers exactly as stated
3. Remove citation brackets like [1], [2] for voice (just state the facts)
4. If Perplexity said it couldn't find something, relay that

DO NOT add any information not in the Perplexity answer.
DO NOT use your own knowledge - only use what Perplexity provided.

Format for natural speech - as if you're telling a friend what you just read."""
```

### 5. Contextual Query Building

```python
def build_contextualized_query(user_text: str, history: list) -> str:
    """
    Add conversation context to ambiguous queries.
    "who won?" → "Texas Tech BYU basketball who won?"
    """
    if not history:
        return user_text

    if len(user_text.split()) < 10:
        last_exchange = history[-1] if history else {}
        last_topic = last_exchange.get("assistant", "")[:200]
        return f"{last_topic} {user_text}"

    return user_text
```

### 6. Search Triggers with Memory Detection

The system must distinguish between **web search questions** and **memory recall questions**:

```python
SEARCH_TRIGGERS = [
    # Explicit search
    "search for", "look up", "find out", "search", "google", "research",

    # Current events
    "what's the latest", "current news", "recent news", "today's", "latest",

    # Factual questions
    "who is", "what is", "who was", "what was", "who are", "what are",
    "who won", "what happened", "when did", "where is", "how much is",

    # Sports
    "score", "game", "match", "stats", "statistics", "points", "goals",
    "nba", "nfl", "mlb", "ncaa", "basketball", "football", "team", "player",

    # Finance
    "stock", "market", "price of", "bitcoin", "crypto", "dow jones",

    # Weather
    "weather", "temperature", "forecast",

    # General
    "tell me about", "give me", "how many", "how much",
]

# CRITICAL: Memory questions should NOT trigger web search
MEMORY_PATTERNS = [
    "first question", "last question", "previous question",
    "did i ask", "did i say", "what did i",
    "i asked you", "i told you", "i said",
    "our conversation", "this conversation",
    "remember when", "earlier i",
    "my name", "who am i",
]

def should_search(text: str) -> bool:
    """Detect if user query needs web search."""
    text_lower = text.lower()

    # Memory questions bypass search - answered from conversation history
    if any(pattern in text_lower for pattern in MEMORY_PATTERNS):
        print(f"[Search] Skipped - memory question detected", flush=True)
        return False

    return any(trigger in text_lower for trigger in SEARCH_TRIGGERS)
```

**Why this matters:** Without memory detection, "What was my first question?" would trigger a web search (because it contains "what was") instead of being answered from conversation history.

---

## Perplexity Models

| Model | Use Case | Speed | Depth |
|-------|----------|-------|-------|
| `sonar` | General queries | Fast (~3s) | Good |
| `sonar-pro` | Complex research | Slower (~5-8s) | Deep |
| `sonar-reasoning` | Multi-step reasoning | Slowest | Deepest |

For voice assistants, `sonar` is the best balance of speed and accuracy.

---

## Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...           # For ASR (Whisper) and Router (8B)
PERPLEXITY_API_KEY=pplx-...    # For search + synthesis
CARTESIA_API_KEY=...           # For TTS

# Recommended (8x faster LLM)
CEREBRAS_API_KEY=...           # Get from https://cloud.cerebras.ai/
                               # Falls back to Groq if not set

# Optional (backup ASR)
ASSEMBLYAI_API_KEY=...
```

---

## Quick Start

```bash
# 1. Navigate to directory
cd liquid-audio/cpu-test

# 2. Activate venv
source ../.venv/bin/activate

# 3. Set environment variables (or use .env file)
export GROQ_API_KEY="your-key"
export PERPLEXITY_API_KEY="your-key"
export CARTESIA_API_KEY="your-key"

# 4. Run
python app_perplexity_stack.py

# 5. Open browser
open http://localhost:5008
```

---

## Comparison: Tavily vs Perplexity

| Aspect | Tavily Stack | Perplexity Stack |
|--------|-------------|------------------|
| **Port** | 5006 | 5008 |
| **Search Latency** | ~2-3s | ~3-5s |
| **Grounding** | LLM interprets raw results | Pre-grounded by Perplexity |
| **Hallucination Risk** | Medium (needs strict prompt) | Low (pre-verified) |
| **Citation Format** | Source URLs in context | [1], [2] inline citations |
| **Best For** | Speed-critical apps | Accuracy-critical apps |

**When to use Tavily:**
- Need fastest possible response
- Simple factual queries
- Cost-sensitive (Tavily is cheaper)

**When to use Perplexity:**
- Accuracy is paramount
- Complex queries requiring synthesis
- Want pre-verified answers

---

## Logs

```bash
# Watch real-time logs
tail -f /tmp/perplexity_stack.log

# Expected output:
# [Search] Triggered for: 'who won the game'
# [Perplexity] Query: 'Texas Tech BYU basketball...' in 3421ms
# [Perplexity] Response length: 847 chars
```

---

## Gapless Audio Playback

Same as Tavily stack - uses Web Audio API with pre-scheduling:

```javascript
function queueAudioChunk(pcmData) {
    const int16 = new Int16Array(
        pcmData.buffer,
        pcmData.byteOffset,
        pcmData.byteLength / 2
    );
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
    }

    const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
    audioBuffer.getChannelData(0).set(float32);

    // Pre-schedule for gapless playback
    if (nextPlayTime < audioContext.currentTime) {
        nextPlayTime = audioContext.currentTime;
    }

    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start(nextPlayTime);

    nextPlayTime += audioBuffer.duration;
}
```

---

## Barge-in (Interrupt Bot While Speaking)

When the user presses the record button, any playing audio stops immediately:

```javascript
async function startRecording() {
    // Barge-in: immediately stop any playing audio when user starts speaking
    resetAudioPlayback();

    initAudioContext();
    // ... rest of recording logic
}

function resetAudioPlayback() {
    // Stop all scheduled sources instantly
    scheduledSources.forEach(source => {
        try { source.stop(); } catch(e) {}
    });
    scheduledSources = [];
    nextPlayTime = 0;
    // ...
}
```

**Why this matters:** Natural conversations have interruptions. Without barge-in, users must wait for the bot to finish speaking before asking follow-up questions.

---

## Debug Logging

Conversation history is logged before each LLM call for debugging:

```
============================================================
[LLM] Conversation history being sent to model:
------------------------------------------------------------
  [0] SYSTEM: You are a voice assistant having a real-time...
  [1] USER: Hello, can you hear me?
  [2] ASSISTANT: Yes, I can hear you! How can I help?
  [3] USER: What was my first question?
------------------------------------------------------------
[LLM] Total messages: 4 (history exchanges: 1)
============================================================

[LLM] Model response (245ms): Your first question was "Hello, can you hear me?"
============================================================
```

This helps verify:
- Conversation history is being maintained
- Memory recall questions receive proper context
- Message count matches expected exchanges

---

## Files

```
cpu-test/
├── app_perplexity_stack.py       # Perplexity version (port 5008)
├── app_api_stack.py              # Tavily version (port 5006)
├── PERPLEXITY-VOICE-ASSISTANT.md # This documentation
├── GROUNDED-VOICE-ASSISTANT.md   # Tavily documentation
├── .env                          # API keys (gitignored)
└── templates/
    └── (inline HTML in app.py)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| "sonar" model not found | Wrong API endpoint | Ensure `base_url="https://api.perplexity.ai"` |
| Slow responses | Using sonar-pro | Switch to `sonar` for speed |
| Missing citations | API version | Citations may be in different response field |
| Rate limited | Too many requests | Add retry logic with backoff |
| Memory questions search web | Missing MEMORY_PATTERNS | Add patterns like "first question", "i asked you" |
| "What did I ask?" fails | Triggers "what" keyword | MEMORY_PATTERNS must be checked BEFORE SEARCH_TRIGGERS |
| Says "I'm text-based" | Persona drift/character breaking | Use SOTA layered prompt: IDENTITY + BEHAVIORAL RULES + STYLE |
| Contradicts itself on memory | Weak persona instructions | Add explicit "NEVER say you don't have history when you do" |
| Can't interrupt bot while speaking | Missing barge-in | Call `resetAudioPlayback()` at start of `startRecording()` |
| No visibility into LLM context | Missing debug logs | Check terminal for `[LLM] Conversation history` output |

---

## Future Improvements

1. **Streaming Perplexity** - Stream tokens as they arrive (if API supports)
2. **Model routing** - Use `sonar` for simple, `sonar-pro` for complex queries
3. **Citation display** - Show sources in UI while speaking answer
4. **Hybrid approach** - Tavily for speed, Perplexity fallback for accuracy
5. **Caching** - Cache common queries to reduce latency

---

*Last updated: January 19, 2026*

---

## Changelog

- **Jan 19, 2026**: 🧠 **Upgraded main LLM to Cerebras Qwen3-32B** - 8x faster (2,400 t/s vs 280 t/s), with automatic fallback to Groq if key not set
- **Jan 19, 2026**: Barge-in feature - audio stops instantly when user presses record button
- **Jan 19, 2026**: Debug logging for conversation history sent to LLM
- **Jan 19, 2026**: SOTA persona consistency prompt - layered IDENTITY/BEHAVIORAL RULES/STYLE structure to prevent character drift (e.g., "I'm text-based")
- **Jan 19, 2026**: Added MEMORY_PATTERNS to prevent memory recall questions from triggering web search
- **Jan 18, 2026**: Initial Perplexity stack implementation
