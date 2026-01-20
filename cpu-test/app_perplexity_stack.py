#!/usr/bin/env python3
"""
Ultra-Low Latency Voice Assistant (Perplexity Search Edition)
Groq Whisper (ASR) + Perplexity (Search) + Cerebras Qwen3-32B (LLM) + Cartesia (TTS)

Architecture:
    - Router: Groq Llama 3.1 8B (fast search detection)
    - Main LLM: Cerebras Qwen3-32B @ 2,400 t/s (or Groq fallback @ 280 t/s)

Target: ~500ms without search, ~4-6s with search
Port: 5008

Usage:
    python app_perplexity_stack.py
    (loads .env file automatically)

Environment:
    CEREBRAS_API_KEY - Get from https://cloud.cerebras.ai/
"""

# Load .env file FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import asyncio
import base64
import io
import json
import os
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# API clients
import assemblyai as aai
from groq import Groq
import httpx
from openai import OpenAI  # Perplexity uses OpenAI-compatible API

app = Flask(__name__)

# =============================================================================
# Configuration
# =============================================================================

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")

# Validate keys
missing_keys = []
if not ASSEMBLYAI_API_KEY:
    missing_keys.append("ASSEMBLYAI_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")
if not CARTESIA_API_KEY:
    missing_keys.append("CARTESIA_API_KEY")
if not PERPLEXITY_API_KEY:
    missing_keys.append("PERPLEXITY_API_KEY")

if missing_keys:
    print(f"\n[ERROR] Missing API keys: {', '.join(missing_keys)}")
    print("Set them with:")
    for key in missing_keys:
        print(f"  export {key}='your-key-here'")
    print()

# Initialize clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Perplexity uses OpenAI-compatible API
perplexity_client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
) if PERPLEXITY_API_KEY else None

# Cerebras client (OpenAI-compatible API) - Main LLM brain (2,400 t/s)
cerebras_client = OpenAI(
    api_key=CEREBRAS_API_KEY,
    base_url="https://api.cerebras.ai/v1"
) if CEREBRAS_API_KEY else None

# Cartesia settings
CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VOICE_ID = "a167e0f3-df7e-4d52-a9c3-f949145efdab"  # Blake - Helpful Agent (energetic adult male)
# Other good voices:
# "bf991597-6c13-47e4-8411-91ec2de5c466"  # Newsman (clear, professional)
# "71a7ad14-091c-4e8e-a314-022ece01c121"  # Friendly Woman

# Audio cache
_audio_cache = {}

# Conversation memory (last 10 exchanges)
conversation_history = []
MAX_HISTORY = 10

# =============================================================================
# Web Search (Perplexity)
# =============================================================================

# Keywords that trigger web search - be aggressive to avoid hallucinations
SEARCH_TRIGGERS = [
    # Explicit search requests
    "search for", "look up", "find out", "search", "google", "look online", "find online",
    "research", "check the web", "find me",
    # Current events
    "what's the latest", "current news", "recent news", "today's", "this week",
    "what is the current", "what are the latest", "news about", "latest",
    # Questions about recent events
    "who won", "what happened", "when did", "where is", "how much is",
    "who is", "what is", "who was", "what was", "who are", "what are",
    # Finance / stocks / prices
    "stock", "market", "price of", "how much does", "cost of",
    "bitcoin", "crypto", "dow jones", "s&p", "nasdaq",
    # Weather
    "weather", "temperature", "forecast",
    # Sports - expanded
    "score", "game", "match", "playoff", "championship", "stats", "statistics",
    "points", "goals", "touchdown", "winner", "loser", "team", "player",
    "nba", "nfl", "mlb", "nhl", "ncaa", "college", "basketball", "football",
    # General factual queries
    "tell me about", "give me", "how many", "how much", "when is", "where is",
]

def should_search(text: str) -> bool:
    """Detect if user query needs web search."""
    text_lower = text.lower()

    # Memory/conversation questions should NOT trigger search
    MEMORY_PATTERNS = [
        "first question", "last question", "previous question",
        "did i ask", "did i say", "what did i",
        "i asked you", "i told you", "i said",
        "our conversation", "this conversation",
        "remember when", "earlier i",
        "my name", "who am i",
    ]

    # If it's a memory question, don't search
    if any(pattern in text_lower for pattern in MEMORY_PATTERNS):
        print(f"[Search] Skipped - memory question detected: '{text[:50]}'", flush=True)
        return False

    return any(trigger in text_lower for trigger in SEARCH_TRIGGERS)


def build_contextualized_query(user_text: str, history: list) -> str:
    """
    Build a search query that includes conversation context.
    This prevents ambiguous queries like "who won?" from returning unrelated results.
    """
    if not history:
        return user_text

    # Get the last 2-3 exchanges for context (most relevant recent context)
    recent_history = history[-3:] if len(history) >= 3 else history

    # Build context summary from recent conversation
    context_parts = []
    for exchange in recent_history:
        # Include key info from both user questions and assistant responses
        context_parts.append(exchange.get("user", ""))
        context_parts.append(exchange.get("assistant", ""))

    context_text = " ".join(context_parts)

    # For short/ambiguous queries, prepend conversation context
    # This turns "who was the high scorer?" into "Texas Tech BYU basketball game who was the high scorer?"
    if len(user_text.split()) < 10:  # Short query likely needs context
        # Extract key entities from context (simple approach: just use last exchange)
        last_exchange = history[-1] if history else {}
        last_topic = last_exchange.get("assistant", "")[:200]  # First 200 chars of last response

        # Build enriched query
        enriched_query = f"{last_topic} {user_text}"
        print(f"[Search] Enriched query: '{user_text}' → '{enriched_query[:100]}...'", flush=True)
        return enriched_query

    return user_text


def web_search(query: str, max_results: int = 5, topic: str = "general") -> tuple[str, int]:
    """
    Search the web using Perplexity's sonar model with built-in search.
    Perplexity does search + synthesis in one call - no separate grounding needed.
    Returns (context_string, latency_ms)
    """
    if not perplexity_client:
        return "", 0

    start = time.time()

    try:
        # Perplexity sonar models have built-in web search
        # They return a synthesized answer with citations
        response = perplexity_client.chat.completions.create(
            model="sonar",  # or "sonar-pro" for deeper research
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Provide factual, well-sourced answers. Include specific data, numbers, and cite your sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=1000,
            temperature=0.2,  # Low for factual accuracy
        )

        # Extract the response content
        content = response.choices[0].message.content

        # Perplexity includes citations in the response
        # Format: The answer text with [1], [2] citations
        # Citations are in response.citations if available

        latency_ms = int((time.time() - start) * 1000)

        # Build context with citations if available
        context_parts = [f"PERPLEXITY ANSWER:\n{content}"]

        # Add citations if present in response
        if hasattr(response, 'citations') and response.citations:
            context_parts.append("\nSOURCES:")
            for i, citation in enumerate(response.citations, 1):
                context_parts.append(f"[{i}] {citation}")

        context = "\n".join(context_parts)

        print(f"[Perplexity] Query: '{query[:60]}...' in {latency_ms}ms", flush=True)
        print(f"[Perplexity] Response length: {len(content)} chars", flush=True)

        return context, latency_ms

    except Exception as e:
        print(f"[Perplexity] Search error: {e}", flush=True)
        return "", int((time.time() - start) * 1000)


# =============================================================================
# Groq Whisper - Speech to Text (FAST)
# =============================================================================

def transcribe_audio_groq_whisper(audio_path: str) -> tuple[str, int]:
    """
    Transcribe audio using Groq's Whisper (whisper-large-v3-turbo).
    Much faster than AssemblyAI batch mode.
    Returns (text, latency_ms)
    """
    start = time.time()

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3-turbo",  # Fastest Whisper on Groq
            response_format="text",
            language="en",
        )

    latency_ms = int((time.time() - start) * 1000)

    return transcription.strip(), latency_ms


# =============================================================================
# AssemblyAI - Speech to Text (BACKUP - slower batch mode)
# =============================================================================

def transcribe_audio_assemblyai(audio_path: str) -> tuple[str, int]:
    """
    Transcribe audio using AssemblyAI.
    Returns (text, latency_ms)
    """
    start = time.time()

    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.nano,  # Fastest model
        language_code="en",
    )

    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    latency_ms = int((time.time() - start) * 1000)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI error: {transcript.error}")

    return transcript.text or "", latency_ms


# =============================================================================
# LLM - Cerebras Qwen3-32B (primary) / Groq Llama 3.3 70B (fallback)
# =============================================================================

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

def check_search_relevance(user_text: str, search_context: str) -> tuple[bool, str]:
    """
    Quick LLM check: Are search results relevant? If not, suggest better query.
    Returns (is_relevant, suggested_query)
    """
    if not search_context:
        return False, user_text

    check_prompt = f"""User asked: "{user_text}"

Search results preview: {search_context[:1000]}

Is this relevant? Reply ONLY with:
- "YES" if results answer the question
- "NO: <better search query>" if results are off-topic

Example: "NO: Texas Tech BYU basketball box score January 2026" """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast small model for this check
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=50,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()

        if result.upper().startswith("YES"):
            return True, ""
        elif result.upper().startswith("NO:"):
            better_query = result[3:].strip()
            print(f"[Search] Results not relevant, retry with: '{better_query}'", flush=True)
            return False, better_query
    except Exception as e:
        print(f"[Search] Relevance check error: {e}", flush=True)

    return True, ""  # Default to using results if check fails


def generate_response_groq(user_text: str) -> tuple[str, int, int]:
    """
    Generate response using Groq's Llama 3.3 70B.
    Includes conversation history for multi-turn memory.
    Performs web search if query triggers search keywords.
    Multi-step: retries with better query if first search misses.
    Returns (response_text, llm_latency_ms, search_latency_ms)
    """
    global conversation_history
    start = time.time()
    search_latency = 0
    search_context = ""

    # Check if web search is needed
    if should_search(user_text):
        print(f"[Search] Triggered for: '{user_text}'", flush=True)

        # Build contextualized query using conversation history
        search_query = build_contextualized_query(user_text, conversation_history)
        search_context, search_latency = web_search(search_query)

        # Multi-step: Check if results are relevant, retry if not
        if search_context:
            is_relevant, better_query = check_search_relevance(user_text, search_context)
            if not is_relevant and better_query:
                print(f"[Search] Retry search with: '{better_query}'", flush=True)
                retry_context, retry_latency = web_search(better_query)
                if retry_context:
                    search_context = retry_context
                    search_latency += retry_latency
    else:
        print(f"[Search] NOT triggered for: '{user_text}'", flush=True)

    # Choose system prompt based on whether we have search results
    system_prompt = SYSTEM_PROMPT_WITH_SEARCH if search_context else SYSTEM_PROMPT

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 10 exchanges)
    for exchange in conversation_history:
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    # Add search context if available
    if search_context:
        user_content = f"[Web Search Results]\n{search_context}\n\n[User Question]\n{user_text}"
    else:
        user_content = user_text

    messages.append({"role": "user", "content": user_content})

    # Log conversation history for debugging
    print("\n" + "="*60, flush=True)
    print("[LLM] Conversation history being sent to model:", flush=True)
    print("-"*60, flush=True)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        # Truncate long content for readability
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"  [{i}] {role}: {content}", flush=True)
    print("-"*60, flush=True)
    print(f"[LLM] Total messages: {len(messages)} (history exchanges: {len(conversation_history)})", flush=True)
    print("="*60 + "\n", flush=True)

    # Lower temperature for factual accuracy (research shows this reduces hallucination)
    # Use 0.3 for search-grounded responses, 0.7 for general chat
    temp = 0.3 if search_context else 0.7

    # Use Cerebras Qwen3-32B (2,400 t/s) if available, fallback to Groq
    if cerebras_client:
        response = cerebras_client.chat.completions.create(
            model="qwen-3-32b",
            messages=messages,
            max_tokens=200,
            temperature=temp,
        )
        model_used = "Cerebras Qwen3-32B"
    else:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=200,
            temperature=temp,
        )
        model_used = "Groq Llama 3.3 70B"

    llm_latency = int((time.time() - start) * 1000) - search_latency
    reply = response.choices[0].message.content.strip()

    # Log model response
    print(f"[LLM] {model_used} response ({llm_latency}ms): {reply}", flush=True)
    print("="*60 + "\n", flush=True)

    # Store in history (keep last MAX_HISTORY exchanges)
    conversation_history.append({"user": user_text, "assistant": reply})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

    return reply, llm_latency, search_latency


# =============================================================================
# Cartesia - Text to Speech (Streaming SSE)
# =============================================================================

def synthesize_speech_cartesia_streaming(text: str, sample_rate: int = 24000):
    """
    Stream speech synthesis using Cartesia SSE.
    Yields (chunk_type, data) tuples:
      - ('audio', raw_pcm_bytes)
      - ('done', total_latency_ms)

    Args:
        text: Text to synthesize
        sample_rate: Output sample rate (24000 for browser, 48000 for test harness)
    """
    start = time.time()
    first_chunk_time = None

    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    payload = {
        "model_id": "sonic-2",
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": CARTESIA_VOICE_ID,
        },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        },
    }

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", "https://api.cartesia.ai/tts/sse", headers=headers, json=payload) as response:
            response.raise_for_status()

            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk

                # Parse SSE events
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)

                    # Parse event
                    event_type = None
                    event_data = None

                    for line in event_str.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()

                    if event_type == "chunk" and event_data:
                        try:
                            data = json.loads(event_data)
                            if "data" in data:
                                audio_bytes = base64.b64decode(data["data"])
                                if first_chunk_time is None:
                                    first_chunk_time = time.time()
                                yield ("audio", audio_bytes)
                        except (json.JSONDecodeError, KeyError):
                            pass

                    elif event_type == "done":
                        total_ms = int((time.time() - start) * 1000)
                        ttfb_ms = int((first_chunk_time - start) * 1000) if first_chunk_time else total_ms
                        yield ("done", {"total_ms": total_ms, "ttfb_ms": ttfb_ms})


def synthesize_speech_cartesia(text: str) -> tuple[bytes, int]:
    """
    Synthesize speech using Cartesia Sonic (non-streaming, for backward compat).
    Returns (wav_bytes, latency_ms)
    """
    start = time.time()

    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json",
    }

    payload = {
        "model_id": "sonic-2",
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": CARTESIA_VOICE_ID,
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(CARTESIA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        wav_bytes = response.content

    latency_ms = int((time.time() - start) * 1000)

    return wav_bytes, latency_ms


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant - API Stack</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 28px;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #00d4ff, #7b2dff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }
        .stack-badge {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .badge {
            background: rgba(255,255,255,0.1);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 500;
        }
        .badge.asr { border: 1px solid #00d4ff; color: #00d4ff; }
        .badge.llm { border: 1px solid #7b2dff; color: #7b2dff; }
        .badge.tts { border: 1px solid #ff6b9d; color: #ff6b9d; }

        .timing-bar {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        .timing-box {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 12px 8px;
            text-align: center;
        }
        .timing-box .label {
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .timing-box .value {
            font-size: 20px;
            font-weight: 600;
            margin-top: 4px;
        }
        .timing-box.asr .value { color: #00d4ff; }
        .timing-box.llm .value { color: #7b2dff; }
        .timing-box.tts .value { color: #ff6b9d; }
        .timing-box.total .value { color: #00ff88; }

        .status {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            margin-bottom: 15px;
            min-height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .record-btn {
            width: 100%;
            padding: 20px;
            font-size: 18px;
            font-weight: 600;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            background: linear-gradient(135deg, #ff6b6b, #ff8e53);
            color: white;
            transition: transform 0.1s, box-shadow 0.1s;
            margin-bottom: 15px;
        }
        .record-btn:active {
            transform: scale(0.98);
            box-shadow: 0 0 30px rgba(255,107,107,0.5);
        }
        .record-btn.recording {
            background: linear-gradient(135deg, #ff4757, #ff3838);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.4); }
            50% { box-shadow: 0 0 20px 10px rgba(255,71,87,0); }
        }

        .response-area {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            min-height: 100px;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        .response-area .label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        audio {
            width: 100%;
            border-radius: 8px;
            margin-top: 10px;
        }

        .cost-display {
            text-align: center;
            font-size: 12px;
            color: #666;
            margin-top: 15px;
        }
        .cost-display span { color: #00ff88; }

        .clear-btn {
            width: 100%;
            padding: 12px;
            font-size: 14px;
            border: 1px solid #444;
            border-radius: 8px;
            cursor: pointer;
            background: transparent;
            color: #888;
            margin-top: 10px;
            transition: all 0.2s;
        }
        .clear-btn:hover {
            border-color: #ff6b6b;
            color: #ff6b6b;
        }
        .memory-status {
            text-align: center;
            font-size: 11px;
            color: #555;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Voice Assistant</h1>
        <p class="subtitle">AssemblyAI + Groq + Cartesia</p>

        <div class="stack-badge">
            <span class="badge asr">ASR: Groq Whisper</span>
            <span class="badge llm">LLM: Cerebras Qwen3-32B</span>
            <span class="badge tts">TTS: Cartesia Sonic</span>
        </div>

        <div class="timing-bar">
            <div class="timing-box asr">
                <div class="label">ASR</div>
                <div class="value" id="asr-time">-</div>
            </div>
            <div class="timing-box llm">
                <div class="label">LLM</div>
                <div class="value" id="llm-time">-</div>
            </div>
            <div class="timing-box tts">
                <div class="label">TTS</div>
                <div class="value" id="tts-time">-</div>
            </div>
            <div class="timing-box total">
                <div class="label">Total</div>
                <div class="value" id="total-time">-</div>
            </div>
        </div>

        <div class="status" id="status">Click and hold to talk</div>

        <button class="record-btn" id="recordBtn"
                onmousedown="startRecording()"
                onmouseup="stopRecording()"
                ontouchstart="startRecording()"
                ontouchend="stopRecording()">
            Hold to Talk
        </button>

        <div class="response-area">
            <div class="label">Response</div>
            <div id="response">Waiting for input...</div>
        </div>

        <audio id="audioPlayer" controls style="display:none;"></audio>

        <div class="cost-display">
            Estimated cost: <span>~$0.034/min</span>
        </div>

        <button class="clear-btn" onclick="clearMemory()">Clear Memory</button>
        <div class="memory-status" id="memoryStatus">Memory: 0/10 exchanges</div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let startTime;

        // Web Audio API for streaming playback - GAPLESS using pre-scheduling
        let audioContext = null;
        let nextPlayTime = 0;
        let chunkCount = 0;
        let totalSamples = 0;
        let scheduledSources = [];

        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }

        function queueAudioChunk(pcmData) {
            chunkCount++;

            // Convert Int16 PCM to Float32 for Web Audio API
            // IMPORTANT: Use explicit byteOffset and length to avoid buffer misalignment
            const int16 = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength / 2);
            const float32 = new Float32Array(int16.length);

            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }

            totalSamples += int16.length;

            const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
            audioBuffer.getChannelData(0).set(float32);

            // PRE-SCHEDULE: Calculate exact start time for gapless playback
            const currentTime = audioContext.currentTime;

            // If this is the first chunk or we've fallen behind, start from now
            if (nextPlayTime < currentTime) {
                nextPlayTime = currentTime;
            }

            // Create source and schedule it to start at exact time
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);

            // Schedule to start at precise time (gapless)
            source.start(nextPlayTime);

            // Track for cleanup
            scheduledSources.push(source);

            // Update next play time to exactly when this buffer ends
            nextPlayTime += audioBuffer.duration;

            // Log every 10th chunk to reduce console spam
            if (chunkCount % 10 === 1) {
                const scheduleAhead = (nextPlayTime - currentTime) * 1000;
                console.log(`[Audio] Chunk ${chunkCount}: ${audioBuffer.duration.toFixed(3)}s, scheduled ${scheduleAhead.toFixed(0)}ms ahead`);
            }
        }

        function resetAudioPlayback() {
            // Stop all scheduled sources
            scheduledSources.forEach(source => {
                try { source.stop(); } catch(e) {}
            });
            scheduledSources = [];
            nextPlayTime = 0;
            chunkCount = 0;
            totalSamples = 0;
            if (audioContext) {
                nextPlayTime = audioContext.currentTime;
            }
            console.log('[Audio] Reset playback state');
        }

        async function startRecording() {
            // Barge-in: immediately stop any playing audio when user starts speaking
            resetAudioPlayback();

            initAudioContext();
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });

                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];

                mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunks, { type: 'audio/webm' });
                    sendAudio(blob);
                    stream.getTracks().forEach(t => t.stop());
                };

                mediaRecorder.start();
                startTime = Date.now();

                document.getElementById('status').textContent = 'Recording...';
                document.getElementById('recordBtn').classList.add('recording');
                document.getElementById('recordBtn').textContent = 'Recording...';

            } catch (err) {
                document.getElementById('status').textContent = 'Microphone access denied';
                console.error(err);
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                document.getElementById('recordBtn').classList.remove('recording');
                document.getElementById('recordBtn').textContent = 'Hold to Talk';
            }
        }

        async function sendAudio(blob) {
            const status = document.getElementById('status');
            const response = document.getElementById('response');
            const audioPlayer = document.getElementById('audioPlayer');

            status.textContent = 'Processing...';
            response.textContent = '';
            audioPlayer.style.display = 'none';
            resetAudioPlayback();

            // Reset timing displays
            ['asr', 'llm', 'tts', 'total'].forEach(id => {
                document.getElementById(id + '-time').textContent = '-';
            });

            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64 = reader.result.split(',')[1];

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ audio: base64 })
                    });

                    const reader2 = res.body.getReader();
                    const decoder = new TextDecoder();
                    let ttfbShown = false;

                    while (true) {
                        const { done, value } = await reader2.read();
                        if (done) break;

                        const text = decoder.decode(value);
                        const lines = text.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const event = JSON.parse(line.slice(6));

                                    if (event.type === 'transcription') {
                                        status.textContent = '"' + event.text + '"';
                                        document.getElementById('asr-time').textContent = event.time_ms + 'ms';

                                    } else if (event.type === 'response') {
                                        response.textContent = event.text;
                                        // Show LLM time, with search time if present
                                        let llmDisplay = event.time_ms + 'ms';
                                        if (event.search_ms) {
                                            llmDisplay += ' +' + event.search_ms + 'ms 🔍';
                                        }
                                        document.getElementById('llm-time').textContent = llmDisplay;
                                        status.textContent = event.search_ms ? 'Searched + streaming...' : 'Streaming audio...';

                                    } else if (event.type === 'audio_start') {
                                        // First audio chunk arriving - show TTFB
                                        if (!ttfbShown) {
                                            document.getElementById('tts-time').textContent = event.ttfb_ms + 'ms TTFB';
                                            ttfbShown = true;
                                        }

                                    } else if (event.type === 'audio_chunk') {
                                        // Decode and queue audio chunk for immediate playback
                                        const pcmBytes = Uint8Array.from(atob(event.data), c => c.charCodeAt(0));
                                        queueAudioChunk(pcmBytes);

                                    } else if (event.type === 'audio_done') {
                                        document.getElementById('tts-time').textContent = event.ttfb_ms + 'ms';
                                        document.getElementById('total-time').textContent = event.total_ms + 'ms';
                                        status.textContent = 'Done in ' + event.total_ms + 'ms (TTFB: ' + event.ttfb_ms + 'ms)';
                                        updateMemoryStatus();

                                    } else if (event.type === 'error') {
                                        status.textContent = 'Error: ' + event.message;
                                        response.textContent = event.message;
                                    }
                                } catch (e) { console.error('Parse error:', e, line); }
                            }
                        }
                    }
                } catch (e) {
                    status.textContent = 'Error: ' + e.message;
                    console.error(e);
                }
            };
            reader.readAsDataURL(blob);
        }

        async function clearMemory() {
            try {
                await fetch('/clear', { method: 'POST' });
                document.getElementById('memoryStatus').textContent = 'Memory: 0/10 exchanges';
                document.getElementById('status').textContent = 'Memory cleared';
                document.getElementById('response').textContent = 'Waiting for input...';
            } catch (e) {
                console.error('Failed to clear memory:', e);
            }
        }

        async function updateMemoryStatus() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                if (data.memory) {
                    document.getElementById('memoryStatus').textContent =
                        'Memory: ' + data.memory.exchanges + '/' + data.memory.max + ' exchanges';
                }
            } catch (e) {}
        }

        // Update memory status on page load
        updateMemoryStatus();
    </script>
</body>
</html>
'''


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/audio/<audio_id>')
def serve_audio(audio_id):
    """Serve cached audio file."""
    if audio_id not in _audio_cache:
        return jsonify({'error': 'Audio not found'}), 404
    wav_data = _audio_cache.pop(audio_id)
    return Response(wav_data, mimetype='audio/wav')


@app.route('/chat', methods=['POST'])
def chat():
    """Process voice input through the API pipeline."""
    import subprocess

    data = request.json
    if not data or 'audio' not in data:
        return jsonify({'error': 'No audio provided'}), 400

    def generate():
        total_start = time.time()

        # 1. Decode and convert audio to WAV
        try:
            audio_bytes = base64.b64decode(data['audio'])

            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(audio_bytes)
                webm_path = f.name

            wav_path = webm_path.replace('.webm', '.wav')
            result = subprocess.run([
                'ffmpeg', '-y', '-i', webm_path,
                '-ar', '16000', '-ac', '1', wav_path
            ], capture_output=True)

            # Check if ffmpeg succeeded
            if result.returncode != 0:
                print(f"[FFmpeg] Error: {result.stderr.decode()}", flush=True)
                yield f"data: {json.dumps({'type': 'error', 'message': f'FFmpeg conversion failed'})}\n\n"
                return

            # Verify wav file exists
            if not os.path.exists(wav_path):
                yield f"data: {json.dumps({'type': 'error', 'message': 'WAV file not created'})}\n\n"
                return

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Audio decode error: {e}'})}\n\n"
            return

        # 2. ASR with Groq Whisper (much faster than AssemblyAI batch)
        try:
            transcription, asr_time = transcribe_audio_groq_whisper(wav_path)
            if not transcription:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No speech detected'})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'transcription', 'text': transcription, 'time_ms': asr_time})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'ASR error: {e}'})}\n\n"
            return
        finally:
            # Cleanup temp files
            try:
                os.unlink(webm_path)
                os.unlink(wav_path)
            except:
                pass

        # 3. LLM with Groq (+ optional web search)
        try:
            reply, llm_time, search_time = generate_response_groq(transcription)
            response_data = {'type': 'response', 'text': reply, 'time_ms': llm_time}
            if search_time > 0:
                response_data['search_ms'] = search_time
            yield f"data: {json.dumps(response_data)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {e}'})}\n\n"
            return

        # 4. TTS with Cartesia (streaming)
        try:
            tts_start = time.time()
            ttfb_ms = None
            total_chunks = 0

            for chunk_type, chunk_data in synthesize_speech_cartesia_streaming(reply):
                if chunk_type == "audio":
                    total_chunks += 1
                    # Send audio chunk as base64 for immediate playback
                    audio_b64 = base64.b64encode(chunk_data).decode('ascii')
                    if ttfb_ms is None:
                        ttfb_ms = int((time.time() - tts_start) * 1000)
                        yield f"data: {json.dumps({'type': 'audio_start', 'ttfb_ms': ttfb_ms, 'sample_rate': 24000})}\n\n"
                    yield f"data: {json.dumps({'type': 'audio_chunk', 'data': audio_b64})}\n\n"

                elif chunk_type == "done":
                    total_time = int((time.time() - total_start) * 1000)
                    tts_time = chunk_data.get("total_ms", 0)
                    yield f"data: {json.dumps({'type': 'audio_done', 'tts_ms': tts_time, 'ttfb_ms': ttfb_ms or 0, 'total_ms': total_time, 'chunks': total_chunks})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'TTS error: {e}'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/health')
def health():
    """Health check endpoint."""
    llm_info = 'Cerebras Qwen3-32B @ 2,400 t/s' if CEREBRAS_API_KEY else 'Groq Llama 3.3 70B @ 280 t/s (fallback)'
    return jsonify({
        'status': 'ok',
        'stack': {
            'asr': 'Groq Whisper',
            'llm': llm_info,
            'router': 'Groq Llama 3.1 8B',
            'tts': 'Cartesia Sonic',
            'search': 'Perplexity Sonar' if PERPLEXITY_API_KEY else 'disabled',
        },
        'keys_configured': {
            'assemblyai': bool(ASSEMBLYAI_API_KEY),
            'groq': bool(GROQ_API_KEY),
            'cartesia': bool(CARTESIA_API_KEY),
            'perplexity': bool(PERPLEXITY_API_KEY),
            'cerebras': bool(CEREBRAS_API_KEY),
        },
        'memory': {
            'exchanges': len(conversation_history),
            'max': MAX_HISTORY,
        },
        'search_triggers': SEARCH_TRIGGERS[:5],  # Show first 5 triggers
    })


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation memory."""
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'cleared', 'message': 'Conversation memory cleared'})


@app.route('/sts', methods=['POST'])
def sts_endpoint():
    """
    STS Test Harness compatible endpoint.
    Accepts: multipart/form-data with 'audio' file
    Returns: SSE stream with {"type": "audio", "data": "base64..."} events
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # MUST save file BEFORE entering generator (Flask request context issue)
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_file.save(f)
            wav_path = f.name
    except Exception as e:
        return jsonify({'error': f'File save error: {e}'}), 500

    def generate():
        total_start = time.time()

        try:
            # 2. ASR with Groq Whisper
            transcription, asr_time = transcribe_audio_groq_whisper(wav_path)
            if not transcription:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No speech detected'})}\n\n"
                return
            # Send text event (test harness expects "text" type)
            yield f"data: {json.dumps({'type': 'text', 'content': transcription})}\n\n"

            # 3. LLM with Groq (+ optional web search)
            reply, llm_time, search_time = generate_response_groq(transcription)
            yield f"data: {json.dumps({'type': 'text', 'content': reply})}\n\n"

            # 4. TTS with Cartesia (streaming at 48kHz for test harness compatibility)
            for chunk_type, data in synthesize_speech_cartesia_streaming(reply, sample_rate=48000):
                if chunk_type == "audio":
                    audio_b64 = base64.b64encode(data).decode('ascii')
                    # Test harness expects {"type": "audio", "data": "..."}
                    yield f"data: {json.dumps({'type': 'audio', 'data': audio_b64})}\n\n"

                elif chunk_type == "done":
                    total_time = int((time.time() - total_start) * 1000)
                    done_data = {'type': 'done', 'total_ms': total_time, 'asr_ms': asr_time, 'llm_ms': llm_time}
                    if search_time > 0:
                        done_data['search_ms'] = search_time
                    yield f"data: {json.dumps(done_data)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            try:
                os.unlink(wav_path)
            except:
                pass

    return Response(generate(), mimetype='text/event-stream')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Voice Assistant - PERPLEXITY EDITION")
    print("  http://localhost:5008")
    print("=" * 60)
    print("\nStack:")
    print("  - ASR: Groq Whisper (Universal)")
    print("  - Search: Perplexity Sonar (built-in synthesis)")
    if CEREBRAS_API_KEY:
        print("  - LLM: Cerebras Qwen3-32B @ 2,400 t/s")
    else:
        print("  - LLM: Groq Llama 3.3 70B @ 280 t/s (fallback)")
    print("  - Router: Groq Llama 3.1 8B (search detection)")
    print("  - TTS: Cartesia (Sonic 2)")
    print(f"  - Search: Perplexity {'[ENABLED]' if PERPLEXITY_API_KEY else '[DISABLED]'}")
    print("\nFeatures:")
    print("  - Conversation memory (last 10 exchanges)")
    print("  - Perplexity search + synthesis (better grounding)")
    print("  - Gapless audio streaming")
    print("\nExpected latency: ~500ms without search, ~4-6s with search")
    print("Note: Perplexity is slower but more accurate than Tavily")
    print("\nAPI Keys:")
    print(f"  - ASSEMBLYAI_API_KEY: {'[SET]' if ASSEMBLYAI_API_KEY else '[MISSING]'}")
    print(f"  - GROQ_API_KEY: {'[SET]' if GROQ_API_KEY else '[MISSING]'}")
    print(f"  - CARTESIA_API_KEY: {'[SET]' if CARTESIA_API_KEY else '[MISSING]'}")
    print(f"  - PERPLEXITY_API_KEY: {'[SET]' if PERPLEXITY_API_KEY else '[MISSING]'}")
    print(f"  - CEREBRAS_API_KEY: {'[SET]' if CEREBRAS_API_KEY else '[NOT SET - using Groq fallback]'}")
    print("=" * 60 + "\n")

    if missing_keys:
        print("[WARNING] Some API keys are missing. Set them before running.\n")

    app.run(host='0.0.0.0', port=5008, debug=False, threaded=True)
