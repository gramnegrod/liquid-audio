"""
Chatbot State Schema for LangGraph Workflow

This module defines the TypedDict state schema used by the LangGraph-based
voice assistant. It's designed for:

1. Two-phase architecture: Context Resolution → Routing → Execution
2. Multi-turn memory with topic persistence
3. Confidence-based routing decisions
4. Semantic caching integration

Based on the research in /docs/Chatbot-primer.md
"""

from typing import TypedDict, Literal, Optional, Annotated
from dataclasses import dataclass, field
import operator
import time


# ============================================================================
# MEMORY TYPES
# ============================================================================

class ConversationTurn(TypedDict):
    """A single turn in the conversation."""
    user: str
    assistant: str
    timestamp: float
    topic: Optional[str]
    search_used: bool


class TopicEntity(TypedDict):
    """Tracked topic/entity for implicit context."""
    name: str
    entity_type: str  # e.g., "person", "company", "product", "concept"
    first_mentioned: float
    last_referenced: float
    mention_count: int


# ============================================================================
# CONFIDENCE TIERS (from calibration experiment)
# ============================================================================

# These thresholds should be validated via calibration_experiment.py
# Default values from industry research (Klarna case study, CoQA paper)
CONFIDENCE_HIGH = 0.90     # Proceed autonomously
CONFIDENCE_MEDIUM = 0.70   # Proceed with monitoring
CONFIDENCE_LOW = 0.40      # Request clarification
# Below LOW = treat as standalone/new topic


ConfidenceTier = Literal["high", "medium", "low", "standalone"]


def get_confidence_tier(confidence: float) -> ConfidenceTier:
    """Map confidence score to action tier."""
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    elif confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    elif confidence >= CONFIDENCE_LOW:
        return "low"
    else:
        return "standalone"


# ============================================================================
# QUERY COMPLEXITY
# ============================================================================

QueryComplexity = Literal["simple", "context_dependent", "complex"]


# ============================================================================
# ROUTE DECISIONS
# ============================================================================

RouteDecision = Literal[
    "direct",      # Answer directly from LLM knowledge
    "search",      # Web search needed for real-time data
    "rag",         # RAG retrieval needed
    "clarify",     # Ask user for clarification
    "datetime",    # Use injected date/time context
    "memory",      # Answer from conversation history
]


# ============================================================================
# MAIN STATE SCHEMA
# ============================================================================

class ChatbotState(TypedDict):
    """
    Complete state for the LangGraph voice assistant workflow.

    This state flows through nodes in this order:
    1. context_detection → Analyze raw query for context markers
    2. context_resolution → Reformulate if needed
    3. query_classifier → Determine complexity
    4. route_decision → Choose execution path
    5. [web_search|rag|direct] → Execute chosen path
    6. synthesis → Generate final response
    7. memory_update → Update conversation history
    """

    # === INPUT ===
    raw_query: str  # Original user query

    # === MEMORY (persisted across turns) ===
    short_term_memory: list[ConversationTurn]  # Last 5 turns
    topic_cache: dict[str, TopicEntity]  # Active topics by name
    session_id: str  # For Redis key namespacing

    # === CONTEXT RESOLUTION (Phase 1) ===
    has_context_markers: bool  # Quick check for pronouns/demonstratives
    standalone_question: Optional[str]  # Reformulated question
    detected_topic: Optional[str]  # Extracted topic entity
    resolution_confidence: float  # 0.0-1.0
    confidence_tier: ConfidenceTier  # Mapped from confidence

    # === REFLECTION (optional, for long-term memory) ===
    reflection: Optional[str]  # Synthesized insight about conversation

    # === ROUTING (Phase 2) ===
    query_complexity: QueryComplexity
    route: RouteDecision
    route_reason: Optional[str]  # Why this route was chosen

    # === SEARCH/RAG RESULTS ===
    search_query: Optional[str]  # Query sent to search
    search_results: Optional[str]  # Raw search response
    search_latency_ms: Optional[int]
    rag_chunks: Optional[list[str]]  # Retrieved RAG chunks

    # === RESPONSE ===
    response: Optional[str]  # Final assistant response
    response_latency_ms: Optional[int]
    citations: list[str]  # Sources cited

    # === FLOW CONTROL ===
    request_id: str  # For barge-in cancellation
    retry_count: int  # For error recovery
    awaiting_clarification: bool  # Waiting for user to clarify
    clarification_question: Optional[str]  # Question asked to user
    is_cancelled: bool  # Barge-in cancellation flag

    # === TIMING ===
    start_time: float
    phase1_latency_ms: Optional[int]
    phase2_latency_ms: Optional[int]
    total_latency_ms: Optional[int]


# ============================================================================
# STATE FACTORY
# ============================================================================

def create_initial_state(
    raw_query: str,
    session_id: str = "default",
    short_term_memory: Optional[list[ConversationTurn]] = None,
    topic_cache: Optional[dict[str, TopicEntity]] = None,
) -> ChatbotState:
    """
    Create initial state for a new query.

    Args:
        raw_query: The user's input query
        session_id: Session identifier for Redis namespacing
        short_term_memory: Previous conversation turns (from Redis)
        topic_cache: Active topics (from Redis)

    Returns:
        ChatbotState ready for LangGraph workflow
    """
    import uuid

    return ChatbotState(
        # Input
        raw_query=raw_query,

        # Memory
        short_term_memory=short_term_memory or [],
        topic_cache=topic_cache or {},
        session_id=session_id,

        # Context Resolution
        has_context_markers=False,
        standalone_question=None,
        detected_topic=None,
        resolution_confidence=0.0,
        confidence_tier="standalone",

        # Reflection
        reflection=None,

        # Routing
        query_complexity="simple",
        route="direct",
        route_reason=None,

        # Search/RAG
        search_query=None,
        search_results=None,
        search_latency_ms=None,
        rag_chunks=None,

        # Response
        response=None,
        response_latency_ms=None,
        citations=[],

        # Flow Control
        request_id=str(uuid.uuid4())[:8],
        retry_count=0,
        awaiting_clarification=False,
        clarification_question=None,
        is_cancelled=False,

        # Timing
        start_time=time.time(),
        phase1_latency_ms=None,
        phase2_latency_ms=None,
        total_latency_ms=None,
    )


# ============================================================================
# STATE HELPERS
# ============================================================================

def add_turn_to_memory(
    state: ChatbotState,
    user_msg: str,
    assistant_msg: str,
    max_turns: int = 5
) -> list[ConversationTurn]:
    """
    Add a conversation turn to short-term memory.

    Returns new memory list (does not mutate state).
    """
    new_turn: ConversationTurn = {
        "user": user_msg,
        "assistant": assistant_msg,
        "timestamp": time.time(),
        "topic": state.get("detected_topic"),
        "search_used": state.get("route") == "search",
    }

    memory = list(state.get("short_term_memory", []))
    memory.append(new_turn)

    # Trim to max_turns
    if len(memory) > max_turns:
        memory = memory[-max_turns:]

    return memory


def update_topic_cache(
    state: ChatbotState,
    topic_name: str,
    entity_type: str = "general"
) -> dict[str, TopicEntity]:
    """
    Update the topic cache with a new or referenced topic.

    Returns new cache dict (does not mutate state).
    """
    cache = dict(state.get("topic_cache", {}))
    now = time.time()

    if topic_name in cache:
        # Update existing
        cache[topic_name] = {
            **cache[topic_name],
            "last_referenced": now,
            "mention_count": cache[topic_name]["mention_count"] + 1,
        }
    else:
        # Add new
        cache[topic_name] = TopicEntity(
            name=topic_name,
            entity_type=entity_type,
            first_mentioned=now,
            last_referenced=now,
            mention_count=1,
        )

    return cache


def get_context_for_reformulation(state: ChatbotState, max_turns: int = 3) -> str:
    """
    Build context string for LLM-based reformulation.

    Returns formatted context from topic cache and recent history.
    """
    parts = []

    # Add current topic from cache
    cache = state.get("topic_cache", {})
    if cache:
        # Get most recently referenced topic
        sorted_topics = sorted(
            cache.items(),
            key=lambda x: x[1].get("last_referenced", 0),
            reverse=True
        )
        if sorted_topics:
            topic = sorted_topics[0]
            parts.append(f"Current topic: {topic[0]}")

    # Add recent conversation history
    memory = state.get("short_term_memory", [])
    if memory:
        recent = memory[-max_turns:]
        history_lines = []
        for turn in recent:
            history_lines.append(f"User: {turn['user']}")
            # Truncate long assistant responses
            assistant_msg = turn['assistant'][:200]
            if len(turn['assistant']) > 200:
                assistant_msg += "..."
            history_lines.append(f"Assistant: {assistant_msg}")
        parts.append("Recent conversation:\n" + "\n".join(history_lines))

    return "\n\n".join(parts)


# ============================================================================
# LINGUISTIC MARKERS
# ============================================================================

# Pronouns that typically indicate reference to prior context
PRONOUNS = {
    "it", "they", "them", "he", "she", "we", "its", "their", "him", "her",
    "his", "hers", "theirs", "ours", "us"
}

# Demonstratives
DEMONSTRATIVES = {"this", "that", "these", "those"}

# Continuation markers
CONTINUATIONS = {
    "the same", "another", "more", "also", "again", "too", "next", "previous",
    "other", "else", "similar"
}

# Implicit reference patterns (definite articles with implicit subject)
IMPLICIT_PATTERNS = [
    "the results", "the answer", "the problem", "the issue", "the solution",
    "the team", "the game", "the score", "the price", "the cost"
]


def has_context_markers(query: str) -> bool:
    """
    Fast heuristic to detect if query likely needs context resolution.

    This is a quick check before calling the LLM for reformulation.
    Based on CoQA research: ~70% of conversational questions have these markers.
    """
    query_lower = query.lower()
    words = set(query_lower.split())

    # Check pronouns
    if words & PRONOUNS:
        return True

    # Check demonstratives
    if words & DEMONSTRATIVES:
        return True

    # Check continuation markers
    for marker in CONTINUATIONS:
        if marker in query_lower:
            return True

    # Check implicit patterns
    for pattern in IMPLICIT_PATTERNS:
        if pattern in query_lower:
            return True

    return False


# ============================================================================
# TYPE EXPORTS
# ============================================================================

__all__ = [
    # Main types
    "ChatbotState",
    "ConversationTurn",
    "TopicEntity",
    "ConfidenceTier",
    "QueryComplexity",
    "RouteDecision",

    # Constants
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",

    # Functions
    "create_initial_state",
    "add_turn_to_memory",
    "update_topic_cache",
    "get_context_for_reformulation",
    "get_confidence_tier",
    "has_context_markers",
]
