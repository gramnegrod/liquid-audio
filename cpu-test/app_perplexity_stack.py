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
import re
import tempfile
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path

import pytz

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# API clients
import assemblyai as aai
from groq import Groq
import httpx
from openai import OpenAI  # Perplexity uses OpenAI-compatible API

# Search router module (v3 - two-phase architecture with barge-in support)
from search_router import (
    init_router, route, RouteDecision,
    cancel_request, get_topic_cache, clear_topic_cache
)

# LangGraph integration (optional - enable with USE_LANGGRAPH=true)
USE_LANGGRAPH = os.environ.get("USE_LANGGRAPH", "").lower() == "true"
_langgraph_workflow = None
_semantic_cache = None

if USE_LANGGRAPH:
    try:
        from chatbot_graph import create_chatbot_graph, run_query_sync, LANGGRAPH_AVAILABLE
        from semantic_cache import get_semantic_cache, CHROMADB_AVAILABLE
        from memory_service import get_memory_service_sync
        print("[LangGraph] LangGraph workflow enabled")
    except ImportError as e:
        print(f"[LangGraph] Import failed: {e}")
        USE_LANGGRAPH = False

# Track active request for barge-in cancellation
_active_request_id: str | None = None

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
if not CEREBRAS_API_KEY:
    missing_keys.append("CEREBRAS_API_KEY")

if missing_keys:
    print(f"\n[ERROR] Missing API keys: {', '.join(missing_keys)}")
    print("Set them with:")
    for key in missing_keys:
        print(f"  export {key}='your-key-here'")
    print()

# Initialize clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Initialize search router with Groq client
if groq_client:
    init_router(groq_client)

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

# Initialize LangGraph workflow if enabled
if USE_LANGGRAPH and groq_client and perplexity_client and cerebras_client:
    try:
        from chatbot_graph import create_chatbot_graph
        from semantic_cache import get_semantic_cache

        def _get_system_prompt():
            """Wrapper for build_system_prompt (defined later)."""
            return build_system_prompt()

        _langgraph_workflow = create_chatbot_graph(
            groq_client=groq_client,
            perplexity_client=perplexity_client,
            cerebras_client=cerebras_client,
            system_prompt_fn=_get_system_prompt,
        )
        _semantic_cache = get_semantic_cache()
        print("[LangGraph] Workflow initialized successfully")
    except Exception as e:
        print(f"[LangGraph] Initialization failed: {e}")
        USE_LANGGRAPH = False
        _langgraph_workflow = None

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
MAX_HISTORY = 5  # Reduced from 10 - too much context confuses simple questions


# =============================================================================
# Dynamic System Prompts with Date/Time Injection
# =============================================================================

def get_current_datetime_cst() -> tuple[str, str]:
    """Get current date and time in Central Time (user's timezone)."""
    cst = pytz.timezone('America/Chicago')
    now = datetime.now(cst)
    date_str = now.strftime("%A, %B %d, %Y")  # e.g., "Tuesday, January 21, 2025"
    time_str = now.strftime("%I:%M %p CST")   # e.g., "3:45 PM CST"
    return date_str, time_str


def build_system_prompt() -> str:
    """Build system prompt with current date/time injected."""
    date_str, time_str = get_current_datetime_cst()
    return f"""You are a voice assistant having a real-time spoken conversation.

IMPORTANT: Respond directly. Do NOT use <think> tags or show your reasoning.

## CURRENT CONTEXT (use this for date/time questions)
Today: {date_str}
Current Time: {time_str}

## RESPONSE RULES
- Maximum 35 words per response (voice output must be concise)
- One main idea per response
- Natural conversational tone
- If unsure, say "I'm not certain about that" - never fabricate

## IDENTITY
- You ARE speaking with the user - they hear your voice
- This is live audio, not text chat

## GROUNDING
- For "what day/date/time" questions: use CURRENT CONTEXT above, don't search
- Never make up facts - if you don't know, say so
- Use conversation history for recall questions"""


def build_search_prompt() -> str:
    """Build search reformatter prompt with current date/time."""
    date_str, time_str = get_current_datetime_cst()
    return f"""You are a voice assistant with access to search results and conversation history.

IMPORTANT: Respond directly. Do NOT use <think> tags.

## CURRENT CONTEXT
Today: {date_str}
Current Time: {time_str}

## CRITICAL: USE CONVERSATION HISTORY
- The user may refer to things mentioned earlier ("the team", "their record", "that game")
- ALWAYS check conversation history to understand what they're referring to
- If they asked about "Texas Tech basketball" earlier and now ask "what's their record", THEY MEAN TEXAS TECH
- Never say "I need more context" if the context is in conversation history

## RESPONSE RULES
- Maximum 50 words (2-3 short sentences)
- Remove citation brackets [1], [2], [3] - just state facts
- Combine search results WITH conversation context to give accurate answers
- If search found nothing useful, say "I couldn't find current information on that"
- Convert UTC times to Central Time if needed"""


# =============================================================================
# Web Search (Perplexity)
# =============================================================================

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
# LLM - Cerebras Qwen3-32B (primary)
# =============================================================================


def generate_response_groq(user_text: str) -> tuple[str, int, int]:
    """
    Generate response using Cerebras Qwen3-32B.
    Includes conversation history for multi-turn memory.
    Smart search routing: only searches for real-time data (weather, prices, etc.)
    Returns (response_text, llm_latency_ms, search_latency_ms)
    """
    global conversation_history
    start = time.time()
    search_latency = 0
    search_context = ""

    # Check if web search is needed (smart routing via search_router v3 - two-phase)
    # Phase 1: Context resolution (uses topic cache + history)
    # Phase 2: Routing classification (SEARCH vs KNOWLEDGE)
    global _active_request_id
    route_result = route(user_text, conversation_history)
    _active_request_id = route_result.request_id  # Track for barge-in

    # Log the resolved context if different from original
    if route_result.resolved_context:
        print(f"[Router] Topic context: '{route_result.resolved_context}'", flush=True)

    if route_result.decision == RouteDecision.SEARCH:
        search_context, search_latency = web_search(route_result.rewritten_query)
        print(f"[Search] '{user_text}' → '{route_result.rewritten_query}'", flush=True)
    elif route_result.decision == RouteDecision.CLARIFY:
        # Low confidence - ask for clarification instead of guessing
        print(f"[Router] CLARIFY needed: {route_result.clarification_question}", flush=True)
        clarify_response = f"I want to make sure I search for the right thing. {route_result.clarification_question} Or could you tell me which team or game you're asking about?"
        # Don't add to history since we're asking for clarification
        _active_request_id = None
        return clarify_response, route_result.latency_ms, 0

    # Choose system prompt based on whether we have search results
    # Dynamic prompts inject current date/time automatically
    system_prompt = build_search_prompt() if search_context else build_system_prompt()

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for exchange in conversation_history:
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    # Add search context if available
    if search_context:
        user_content = f"[Web Search Results]\n{search_context}\n\n[User Question]\n{user_text}"
    else:
        user_content = user_text

    messages.append({"role": "user", "content": user_content})

    # Log for debugging
    print("\n" + "="*60, flush=True)
    print(f"[LLM] System prompt date: {get_current_datetime_cst()[0]}", flush=True)
    print(f"[LLM] Search used: {'Yes' if search_context else 'No'}", flush=True)
    print(f"[LLM] History exchanges: {len(conversation_history)}", flush=True)
    print("="*60 + "\n", flush=True)

    # Lower temperature for factual accuracy
    # Use 0.3 for search-grounded responses, 0.5 for general chat (was 0.7)
    temp = 0.3 if search_context else 0.5

    # Cerebras Qwen3-32B @ 2,400 t/s
    response = cerebras_client.chat.completions.create(
        model="qwen-3-32b",
        messages=messages,
        max_tokens=150,  # Reduced from 200 for shorter voice responses
        temperature=temp,
    )

    llm_latency = int((time.time() - start) * 1000) - search_latency
    reply = response.choices[0].message.content.strip()

    # Strip Qwen thinking tags - model outputs <think>...</think> reasoning
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)  # Closed tags
    reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL)  # Unclosed tags
    reply = reply.strip()

    # Log model response
    print(f"[LLM] Cerebras Qwen3-32B response ({llm_latency}ms): {reply}", flush=True)
    print("="*60 + "\n", flush=True)

    # Store in history (keep last MAX_HISTORY exchanges)
    conversation_history.append({"user": user_text, "assistant": reply})
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

    return reply, llm_latency, search_latency


def generate_response_langgraph(user_text: str, session_id: str = "default") -> tuple[str, int, int]:
    """
    Generate response using LangGraph workflow.

    This is an alternative to generate_response_groq that uses the LangGraph
    state machine for better observability and structured execution.

    Enable with: USE_LANGGRAPH=true

    Returns (response_text, llm_latency_ms, search_latency_ms)
    """
    global conversation_history, _langgraph_workflow, _semantic_cache

    if not _langgraph_workflow:
        print("[LangGraph] Workflow not initialized, falling back to standard router")
        return generate_response_groq(user_text)

    start = time.time()

    # Check semantic cache first
    if _semantic_cache:
        cached = _semantic_cache.get(user_text)
        if cached:
            print(f"[LangGraph] Cache HIT: '{user_text[:40]}...'")
            # Still update conversation history for context
            conversation_history.append({"user": user_text, "assistant": cached.response})
            if len(conversation_history) > MAX_HISTORY:
                conversation_history.pop(0)
            return cached.response, cached.latency_ms, 0

    # Convert conversation_history to LangGraph format
    from chatbot_state import ConversationTurn
    short_term_memory = [
        ConversationTurn(
            user=turn["user"],
            assistant=turn["assistant"],
            timestamp=time.time(),
            topic=None,
            search_used=False,
        )
        for turn in conversation_history
    ]

    try:
        # Run through LangGraph workflow
        from chatbot_graph import run_query_sync
        result = run_query_sync(
            graph=_langgraph_workflow,
            query=user_text,
            session_id=session_id,
            short_term_memory=short_term_memory,
        )

        reply = result.get("response", "I'm sorry, I couldn't generate a response.")
        search_latency = result.get("search_latency_ms", 0)
        llm_latency = result.get("response_latency_ms", 0)
        total_latency = result.get("total_latency_ms", int((time.time() - start) * 1000))

        # Log workflow path
        route = result.get("route", "unknown")
        confidence = result.get("resolution_confidence", 0)
        print(f"[LangGraph] Route: {route}, Confidence: {confidence:.2f}, Total: {total_latency}ms")

        # Cache the response if it was a search
        if _semantic_cache and route == "search":
            _semantic_cache.set(
                query=user_text,
                response=reply,
                route=route,
                latency_ms=total_latency,
            )

        # Update conversation history
        conversation_history.append({"user": user_text, "assistant": reply})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)

        return reply, llm_latency, search_latency

    except Exception as e:
        print(f"[LangGraph] Error: {e}, falling back to standard router")
        return generate_response_groq(user_text)


def generate_response(user_text: str, session_id: str = "default") -> tuple[str, int, int]:
    """
    Main response generator - dispatches to LangGraph or standard router.

    Uses LangGraph if USE_LANGGRAPH=true, otherwise uses standard router.
    """
    if USE_LANGGRAPH and _langgraph_workflow:
        return generate_response_langgraph(user_text, session_id)
    else:
        return generate_response_groq(user_text)


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
        "Cartesia-Version": "2025-04-16",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    payload = {
        "model_id": "sonic-3",
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

    print(f"[TTS] Calling Cartesia SSE with model_id: {payload['model_id']}, voice: {payload['voice']['id']}", flush=True)

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", "https://api.cartesia.ai/tts/sse", headers=headers, json=payload) as response:
            if response.status_code != 200:
                error_body = response.read().decode()
                print(f"[TTS] ERROR {response.status_code}: {error_body[:500]}", flush=True)
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
        "Cartesia-Version": "2025-04-16",
        "Content-Type": "application/json",
    }

    payload = {
        "model_id": "sonic-3",
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
            reply, llm_time, search_time = generate_response(transcription)
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
    return jsonify({
        'status': 'ok',
        'stack': {
            'asr': 'Groq Whisper',
            'llm': 'Cerebras Qwen3-32B @ 2,400 t/s',
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
        'search_routing': 'smart (date/time skip, real-time data search)',
    })


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation memory and topic cache."""
    global conversation_history
    conversation_history = []
    clear_topic_cache()  # Also clear the topic cache
    return jsonify({'status': 'cleared', 'message': 'Conversation memory and topic cache cleared'})


@app.route('/cancel', methods=['POST'])
def cancel_active_request():
    """
    Cancel the active request (for barge-in handling).
    Call this when the user starts speaking to interrupt the current response.
    """
    global _active_request_id
    if _active_request_id:
        cancelled = cancel_request(_active_request_id)
        _active_request_id = None
        return jsonify({
            'status': 'cancelled' if cancelled else 'not_found',
            'message': 'Active request cancelled' if cancelled else 'No active request found'
        })
    return jsonify({'status': 'no_request', 'message': 'No active request to cancel'})


@app.route('/topic', methods=['GET'])
def get_topic():
    """Get the current topic cache state (for debugging)."""
    cache = get_topic_cache()
    return jsonify({
        'topic': cache.topic,
        'entity_type': cache.entity_type,
        'confidence': cache.confidence,
        'age_seconds': time.time() - cache.last_updated if cache.topic else None
    })


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
            reply, llm_time, search_time = generate_response(transcription)
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


@app.route('/sts-sync', methods=['POST'])
def sts_sync_endpoint():
    """
    Synchronous STS endpoint for test harness.
    Accepts: multipart/form-data with 'audio' file
    Returns: JSON with {audio: base64, text: str, ttfa_ms: int}
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_file.save(f)
            wav_path = f.name
    except Exception as e:
        return jsonify({'error': f'File save error: {e}'}), 500

    try:
        total_start = time.time()

        # 1. ASR with Groq Whisper
        transcription, asr_time = transcribe_audio_groq_whisper(wav_path)
        if not transcription:
            return jsonify({'error': 'No speech detected'}), 400

        # 2. LLM with Cerebras
        reply, llm_time, search_time = generate_response_groq(transcription)

        # 3. TTS with Cartesia (collect all chunks)
        audio_chunks = []
        ttfs_ms = None  # Time to First Speech (from request start)
        tts_start = time.time()

        for chunk_type, data in synthesize_speech_cartesia_streaming(reply, sample_rate=48000):
            if chunk_type == "audio":
                if ttfs_ms is None:
                    # TRUE Time to First Speech = from request start to first audio chunk
                    ttfs_ms = int((time.time() - total_start) * 1000)
                audio_chunks.append(data)

        # Combine audio chunks and add WAV header
        raw_pcm = b''.join(audio_chunks)
        wav_bytes = _add_wav_header(raw_pcm, sample_rate=48000)

        total_time = int((time.time() - total_start) * 1000)

        # Calculate TTS first chunk time
        tts_first_ms = ttfs_ms - asr_time - llm_time - (search_time if search_time else 0)

        response_data = {
            'audio': base64.b64encode(wav_bytes).decode('ascii'),
            'text': reply,
            'ttfs_ms': ttfs_ms,  # Time to First Speech (the metric that matters!)
            'total_ms': total_time,
            'asr_ms': asr_time,
            'llm_ms': llm_time,
            'tts_first_ms': tts_first_ms,
        }

        # Add search time if Perplexity was used
        if search_time and search_time > 0:
            response_data['search_ms'] = search_time

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass


def _add_wav_header(pcm_data: bytes, sample_rate: int = 48000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Add WAV header to raw PCM data."""
    import struct
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size (16 for PCM)
        1,   # AudioFormat (1 for PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header + pcm_data


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
    print("  - LLM: Cerebras Qwen3-32B @ 2,400 t/s")
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
    print(f"  - CEREBRAS_API_KEY: {'[SET]' if CEREBRAS_API_KEY else '[MISSING]'}")
    print("=" * 60 + "\n")

    if missing_keys:
        print("[WARNING] Some API keys are missing. Set them before running.\n")

    app.run(host='0.0.0.0', port=5008, debug=False, threaded=True)
