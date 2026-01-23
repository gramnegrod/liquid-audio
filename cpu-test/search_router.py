"""
Search Router Module v3 - Two-Phase Architecture
(GPT-5.2 Architect Recommended)

Key improvements over v2:
- TWO-PHASE PIPELINE: Context Resolution → Routing (separated concerns)
- Topic/entity cache for implicit context carryover
- Request ID + cancellation for barge-in handling
- 5-turn history window
- Full query REWRITING (not prepending)

Usage:
    from search_router import init_router, route, RouteDecision, cancel_request

    init_router(groq_client)
    result = route("What's their record?", conversation_history)

    if result.decision == RouteDecision.SEARCH:
        search_context = web_search(result.rewritten_query)
    elif result.decision == RouteDecision.CLARIFY:
        # Ask user: result.clarification_question

    # For barge-in:
    cancel_request(result.request_id)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time
import json
import re
import uuid

# Groq client - initialized by main app
_groq_client = None

# Confidence threshold - below this, ask clarification
CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# Topic/Entity Cache - Persists context across turns
# =============================================================================

@dataclass
class TopicCache:
    """Tracks active topic/entity for implicit context resolution."""
    topic: Optional[str] = None           # e.g., "the current topic of conversation"
    entity_type: Optional[str] = None     # e.g., "sports_team", "person", "place"
    last_updated: float = 0.0
    confidence: float = 0.0

    def update(self, topic: str, entity_type: str = "general", confidence: float = 0.9):
        """Update the active topic."""
        self.topic = topic
        self.entity_type = entity_type
        self.last_updated = time.time()
        self.confidence = confidence
        print(f"[TopicCache] Updated: '{topic}' ({entity_type})", flush=True)

    def get(self) -> Optional[str]:
        """Get current topic if still valid (within 5 minutes)."""
        if self.topic and (time.time() - self.last_updated) < 300:
            return self.topic
        return None

    def clear(self):
        """Clear the topic cache."""
        self.topic = None
        self.entity_type = None
        self.confidence = 0.0
        print("[TopicCache] Cleared", flush=True)

# Global topic cache
_topic_cache = TopicCache()

# =============================================================================
# Request Cancellation - For barge-in handling
# =============================================================================

# Active requests that can be cancelled
_active_requests: Dict[str, bool] = {}

def cancel_request(request_id: str) -> bool:
    """Cancel an active request (for barge-in)."""
    if request_id in _active_requests:
        _active_requests[request_id] = True  # Mark as cancelled
        print(f"[Router] Cancelled request: {request_id}", flush=True)
        return True
    return False

def is_cancelled(request_id: str) -> bool:
    """Check if a request has been cancelled."""
    return _active_requests.get(request_id, False)

def _create_request_id() -> str:
    """Create a new request ID."""
    request_id = str(uuid.uuid4())[:8]
    _active_requests[request_id] = False
    return request_id

def _cleanup_request(request_id: str):
    """Remove request from active tracking."""
    _active_requests.pop(request_id, None)


class RouteDecision(Enum):
    """Possible routing decisions for a query."""
    SEARCH = "search"        # Needs web search for real-time data
    MEMORY = "memory"        # Answer from conversation history
    KNOWLEDGE = "knowledge"  # Answer from LLM's knowledge
    DATETIME = "datetime"    # Answer using injected date/time context
    CLARIFY = "clarify"      # Need to ask user for clarification


@dataclass
class RouteResult:
    """Result from routing a query."""
    decision: RouteDecision
    rewritten_query: str
    original_query: str
    confidence: float
    latency_ms: int
    clarification_question: Optional[str] = None
    request_id: str = ""                    # For cancellation
    resolved_context: Optional[str] = None  # What Phase 1 resolved


# =============================================================================
# Pattern matchers (fast path - no LLM call needed)
# =============================================================================

DATETIME_PATTERNS = [
    "what day", "what date", "what time", "what's the date",
    "what's the time", "current date", "current time", "day is it",
    "time is it", "date today", "today's date", "time now", "what year"
]

MEMORY_PATTERNS = [
    "first question", "last question", "previous question",
    "did i ask", "did i say", "what did i", "i asked you",
    "i told you", "i said", "our conversation", "this conversation",
    "remember when", "earlier i", "my name", "who am i"
]

# Pronouns and patterns that indicate reference to previous context
AMBIGUOUS_PATTERNS = [
    "their", "they", "them", "it", "its", "that", "those",
    "the team", "the game", "the score", "the record",
    "the girls", "the boys", "the men", "the women",
    "what about", "how about", "and the", "same", "other",
    # Follow-up indicators
    "like the", "like men", "like women", "not the", "no the",
    "i meant", "i mean", "actually", "instead",
]

# Regex patterns for more complex ambiguity detection (e.g., "the basketball game")
AMBIGUOUS_REGEX_PATTERNS = [
    r"\bthe\b.*\bgame\b",      # "the basketball game", "the last basketball game"
    r"\bthe\b.*\bteam\b",      # "the basketball team", "the women's team"
    r"\bthe\b.*\bscore\b",     # "the final score", "the game score"
    r"\bthe\b.*\brecord\b",    # "the season record", "the school record"
    r"\bwho won\b",            # "who won" anywhere (implies context needed)
    r"\bhow did\b.*\bdo\b",    # "how did they do", "how did the team do"
]

# State tracking for CLARIFY flow
_pending_clarification = None  # Stores original query when we ask for clarification


# =============================================================================
# Public API
# =============================================================================

def init_router(client) -> None:
    """
    Initialize router with Groq client.

    Must be called before using route().

    Args:
        client: Groq client instance for LLM calls
    """
    global _groq_client
    _groq_client = client
    print("[Router] Initialized with Groq client", flush=True)


def get_topic_cache() -> TopicCache:
    """Get the current topic cache (for debugging/inspection)."""
    return _topic_cache


def clear_topic_cache() -> None:
    """Clear the topic cache (e.g., on conversation reset)."""
    _topic_cache.clear()


def route(query: str, history: list) -> RouteResult:
    """
    Main entry point for search routing.

    TWO-PHASE ARCHITECTURE (v3):
    - Phase 1: Context Resolution (resolve pronouns, use topic cache)
    - Phase 2: Routing/Classification (SEARCH vs KNOWLEDGE vs etc.)

    Args:
        query: User's query text
        history: List of {"user": str, "assistant": str} conversation exchanges

    Returns:
        RouteResult with:
        - decision: SEARCH, MEMORY, KNOWLEDGE, DATETIME, or CLARIFY
        - rewritten_query: Standalone query for search (if SEARCH)
        - confidence: 0.0-1.0
        - clarification_question: Optional "Did you mean X?" (if CLARIFY)
        - request_id: For cancellation on barge-in
    """
    global _pending_clarification
    start = time.time()
    request_id = _create_request_id()

    try:
        # === CHECK FOR PENDING CLARIFICATION ===
        # If we asked for clarification and user responded, combine them
        if _pending_clarification and len(query.split()) <= 5:
            original_query = _pending_clarification
            _pending_clarification = None

            # User's answer becomes the topic
            clean_answer = query.lower().replace("please", "").replace("actually", "").strip(" .,!")

            # Update topic cache with the clarified entity
            _topic_cache.update(clean_answer, "clarified_entity", 0.95)

            # Combine into natural search query
            combined_query = f"{clean_answer} {original_query.replace('who won', 'result').replace('?', '').replace('.', '')}"
            print(f"[Router] CLARIFY answered: '{query}' + '{original_query}' → '{combined_query}'", flush=True)

            return RouteResult(
                decision=RouteDecision.SEARCH,
                rewritten_query=combined_query,
                original_query=query,
                confidence=0.9,
                latency_ms=int((time.time() - start) * 1000),
                request_id=request_id,
                resolved_context=clean_answer
            )

        # Clear stale clarification if user asked full new question
        if _pending_clarification and len(query.split()) > 5:
            print(f"[Router] Clearing stale clarification (user asked new question)", flush=True)
            _pending_clarification = None

        # === FAST PATH: Skip LLM entirely for known patterns ===
        fast_result = _fast_path(query)
        if fast_result:
            fast_result.latency_ms = int((time.time() - start) * 1000)
            fast_result.request_id = request_id
            return fast_result

        # Check for cancellation
        if is_cancelled(request_id):
            print(f"[Router] Request {request_id} cancelled (pre-phase1)", flush=True)
            return _cancelled_result(query, request_id, start)

        # === PHASE 1: CONTEXT RESOLUTION ===
        # Separate LLM call to resolve pronouns/references using history + topic cache
        resolved_query, resolution_confidence, resolved_topic = _phase1_resolve_context(query, history)

        # Update topic cache if we resolved something new
        if resolved_topic and resolution_confidence > 0.7:
            _topic_cache.update(resolved_topic, "extracted", resolution_confidence)

        # Check for cancellation between phases
        if is_cancelled(request_id):
            print(f"[Router] Request {request_id} cancelled (post-phase1)", flush=True)
            return _cancelled_result(query, request_id, start)

        # === PHASE 2: ROUTING/CLASSIFICATION ===
        # Now classify the resolved query
        needs_search, decision, route_confidence = _phase2_classify(resolved_query)

        # Combined confidence
        final_confidence = min(resolution_confidence, route_confidence)

        result = RouteResult(
            decision=decision,
            rewritten_query=resolved_query,
            original_query=query,
            confidence=final_confidence,
            latency_ms=int((time.time() - start) * 1000),
            request_id=request_id,
            resolved_context=resolved_topic
        )

        # === CONFIDENCE CHECK: Ask clarification if too low ===
        if final_confidence < CONFIDENCE_THRESHOLD and decision == RouteDecision.SEARCH:
            result.decision = RouteDecision.CLARIFY
            result.clarification_question = f"Did you mean: {resolved_query}?"
            _pending_clarification = query
            print(f"[Router] Low confidence ({final_confidence:.2f}) - asking clarification", flush=True)

        return result

    finally:
        _cleanup_request(request_id)


def _cancelled_result(query: str, request_id: str, start: float) -> RouteResult:
    """Create a result for a cancelled request."""
    return RouteResult(
        decision=RouteDecision.KNOWLEDGE,  # Safe fallback
        rewritten_query=query,
        original_query=query,
        confidence=0.0,
        latency_ms=int((time.time() - start) * 1000),
        request_id=request_id,
        resolved_context=None
    )


# =============================================================================
# Internal functions
# =============================================================================

def _fast_path(query: str) -> Optional[RouteResult]:
    """
    Check for patterns that don't need LLM classification.

    Returns RouteResult if fast path matches, None otherwise.
    """
    query_lower = query.lower().strip()

    # Date/time - use injected context (no search needed)
    if any(p in query_lower for p in DATETIME_PATTERNS):
        print(f"[Router] DATETIME (fast path): '{query[:40]}'", flush=True)
        return RouteResult(
            decision=RouteDecision.DATETIME,
            rewritten_query=query,
            original_query=query,
            confidence=1.0,
            latency_ms=0
        )

    # Memory questions - use conversation history
    if any(p in query_lower for p in MEMORY_PATTERNS):
        print(f"[Router] MEMORY (fast path): '{query[:40]}'", flush=True)
        return RouteResult(
            decision=RouteDecision.MEMORY,
            rewritten_query=query,
            original_query=query,
            confidence=1.0,
            latency_ms=0
        )

    # Very short (greetings, acknowledgments)
    if len(query.split()) <= 2:
        print(f"[Router] SHORT (fast path): '{query}'", flush=True)
        return RouteResult(
            decision=RouteDecision.KNOWLEDGE,
            rewritten_query=query,
            original_query=query,
            confidence=0.9,
            latency_ms=0
        )

    return None


def _needs_rewriting(query: str) -> bool:
    """
    Check if query has ambiguous references that need rewriting.

    Returns True if query contains pronouns or references to previous context.
    """
    query_lower = query.lower()

    # Check simple substring patterns
    if any(ref in query_lower for ref in AMBIGUOUS_PATTERNS):
        return True

    # Check regex patterns (e.g., "the basketball game")
    for pattern in AMBIGUOUS_REGEX_PATTERNS:
        if re.search(pattern, query_lower):
            print(f"[Router] Regex matched '{pattern}' in '{query[:40]}'", flush=True)
            return True

    return False


# =============================================================================
# Phase 1: Context Resolution
# =============================================================================

def _phase1_resolve_context(query: str, history: list) -> tuple[str, float, Optional[str]]:
    """
    PHASE 1: Resolve context using topic cache and conversation history.

    ALWAYS uses LLM if any context exists. No pattern matching shortcuts.

    Returns: (resolved_query, confidence, extracted_topic)
    """
    cached_topic = _topic_cache.get()
    recent_history = history[-5:] if history else []

    # No context at all - query is standalone
    if not cached_topic and not recent_history:
        print(f"[Phase1] No context, standalone: '{query[:50]}'", flush=True)
        return query, 1.0, None

    # Build context for LLM
    context_parts = []
    if cached_topic:
        context_parts.append(f"Current topic: {cached_topic}")

    if recent_history:
        history_text = "\n".join([
            f"User: {h.get('user', '')}\nAssistant: {h.get('assistant', '')[:100]}..."
            for h in recent_history
        ])
        context_parts.append(f"Recent conversation:\n{history_text}")

    context = "\n\n".join(context_parts)

    # LLM call for context resolution ONLY (not classification)
    # Domain-agnostic prompt based on CANARD research (Elgohary et al., EMNLP 2019)
    prompt = f"""Given a chat history and the latest user question, formulate a standalone
question which can be understood without the chat history.

CHAT HISTORY:
{context}

LATEST QUESTION: "{query}"

INSTRUCTIONS:
1. If the question contains references that depend on chat history, rewrite it to be self-contained
2. If the question is already standalone or introduces a new topic, return it unchanged
3. Do NOT answer the question - only reformulate if needed

REFORMULATION TRIGGERS (linguistic markers):
- Pronouns: it, they, them, he, she, we, its, their
- Demonstratives: this, that, these, those
- Definite articles with implicit reference: "the results", "the answer", "the price"
- Continuations: the same, another, more, also, again, next, previous
- Ellipsis: questions missing subject/object that chat history provides

CONFIDENCE CALIBRATION:
- 0.90-1.00: Clear linguistic marker present, reformulation certain
- 0.70-0.89: Probable reference to chat history
- 0.40-0.69: Ambiguous, could be standalone
- 0.00-0.39: Standalone question or completely new topic

OUTPUT JSON ONLY:
{{"standalone_question": "the reformulated question", "topic": "main entity or null", "confidence": 0.0-1.0}}"""

    try:
        start = time.time()
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        latency = int((time.time() - start) * 1000)

        # Parse JSON response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            # Support both old "resolved" and new "standalone_question" field names
            resolved = data.get("standalone_question", data.get("resolved", query))
            topic = data.get("topic")
            confidence = float(data.get("confidence", 0.7))

            print(f"[Phase1] Resolved '{query}' → '{resolved}' (topic: {topic}, conf: {confidence:.2f}, {latency}ms)", flush=True)
            return resolved, confidence, topic

    except Exception as e:
        print(f"[Phase1] Resolution failed: {e}", flush=True)

    # Fallback: try using topic cache directly
    if cached_topic:
        resolved = _inject_topic_into_query(query, cached_topic)
        return resolved, 0.6, cached_topic

    return query, 0.4, None


def _inject_topic_into_query(query: str, topic: str) -> str:
    """Simple topic injection for when we have cache but no LLM."""
    query_lower = query.lower()

    # Replace common pronouns
    replacements = [
        ("their ", f"{topic}'s "),
        ("they ", f"{topic} "),
        ("them ", f"{topic} "),
        ("the team", topic),
        ("the game", f"{topic} game"),
    ]

    result = query
    for old, new in replacements:
        if old in query_lower:
            result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
            break

    return result


# =============================================================================
# Phase 2: Routing/Classification
# =============================================================================

def _phase2_classify(query: str) -> tuple[bool, RouteDecision, float]:
    """
    PHASE 2: Classify whether the resolved query needs web search.

    This operates on the RESOLVED query from Phase 1, not the original.

    Returns (needs_search, decision, confidence)
    """
    prompt = f"""Does this query need REAL-TIME web data, or is general knowledge enough?

Query: "{query}"

SEARCH needed for: current events, sports scores, weather, stock prices, recent news, how someone/team is doing, current standings, recent games
KNOWLEDGE sufficient for: definitions, explanations, historical facts, how-to questions, general concepts

Reply with ONLY: SEARCH or KNOWLEDGE"""

    try:
        start = time.time()
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().upper()
        latency = int((time.time() - start) * 1000)

        needs_search = "SEARCH" in result
        decision = RouteDecision.SEARCH if needs_search else RouteDecision.KNOWLEDGE

        print(f"[Phase2] Classified '{query[:40]}' as {result} ({latency}ms)", flush=True)
        return needs_search, decision, 0.9

    except Exception as e:
        print(f"[Phase2] Classification failed: {e}", flush=True)
        return True, RouteDecision.SEARCH, 0.5  # Default to search on error


# Legacy function - kept for compatibility but uses new phases
def _llm_classify(query: str) -> tuple[bool, RouteDecision, float]:
    """
    Use Groq Llama 3.1 8B to classify if query needs web search.

    NOTE: This is now a wrapper around _phase2_classify for backwards compatibility.

    Returns (needs_search, decision, confidence)
    """
    return _phase2_classify(query)


# =============================================================================
# DEPRECATED: Legacy merged function (v2) - kept for reference only
# =============================================================================

def _merged_rewrite_and_classify(query: str, history: list) -> RouteResult:
    """
    DEPRECATED: Use _phase1_resolve_context() + _phase2_classify() instead.

    This v2 merged approach was replaced by the two-phase architecture (v3)
    which separates context resolution from routing classification.

    Keeping for backwards compatibility but route() no longer calls this.
    """
    print("[Router] WARNING: _merged_rewrite_and_classify is deprecated, use two-phase approach", flush=True)

    # Delegate to new two-phase approach
    resolved_query, confidence, topic = _phase1_resolve_context(query, history)
    needs_search, decision, route_confidence = _phase2_classify(resolved_query)

    return RouteResult(
        decision=decision,
        rewritten_query=resolved_query,
        original_query=query,
        confidence=min(confidence, route_confidence),
        latency_ms=0,
        resolved_context=topic
    )
