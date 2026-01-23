"""
LangGraph Voice Assistant Workflow

A state machine implementation for the two-phase voice assistant architecture:
1. Context Resolution (Phase 1) - Reformulate queries using conversation history
2. Routing (Phase 2) - Decide execution path (search, direct, clarify)
3. Execution - Run the chosen path
4. Memory Update - Persist conversation state

Based on: "Generative Agents" (Park et al., 2023) memory architecture
          LangGraph StateGraph patterns

Usage:
    from chatbot_graph import create_chatbot_graph, run_query

    graph = create_chatbot_graph(groq_client, perplexity_client, cerebras_client)
    result = await run_query(graph, "What's their stock price?", session_id="user123")
"""

import time
import json
import re
from typing import Optional, Callable, Any
from functools import partial

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[WARNING] LangGraph not installed. Run: pip install langgraph")

# Local imports
from chatbot_state import (
    ChatbotState,
    create_initial_state,
    add_turn_to_memory,
    update_topic_cache,
    get_context_for_reformulation,
    get_confidence_tier,
    has_context_markers,
    ConfidenceTier,
    RouteDecision,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
)


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def context_detection_node(state: ChatbotState) -> dict:
    """
    Fast heuristic check for context markers (pronouns, demonstratives).

    This is a quick filter before the more expensive LLM call.
    Based on CoQA research: ~70% of conversational questions have these markers.
    """
    query = state["raw_query"]
    has_markers = has_context_markers(query)

    return {
        "has_context_markers": has_markers,
    }


def context_resolution_node(state: ChatbotState, groq_client) -> dict:
    """
    Phase 1: Resolve context using LLM.

    Reformulates queries with pronouns/references into standalone questions.
    Uses the domain-agnostic prompt from search_router.py.
    """
    start = time.time()
    query = state["raw_query"]

    # Get context for reformulation
    context = get_context_for_reformulation(state, max_turns=3)

    # If no context and no markers, skip LLM call
    if not context and not state.get("has_context_markers", False):
        return {
            "standalone_question": query,
            "detected_topic": None,
            "resolution_confidence": 1.0,
            "confidence_tier": "standalone",
            "phase1_latency_ms": int((time.time() - start) * 1000),
        }

    # Domain-agnostic reformulation prompt
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
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            standalone = data.get("standalone_question", query)
            topic = data.get("topic")
            confidence = float(data.get("confidence", 0.7))

            latency = int((time.time() - start) * 1000)
            print(f"[Phase1] '{query}' → '{standalone}' (conf: {confidence:.2f}, {latency}ms)")

            return {
                "standalone_question": standalone,
                "detected_topic": topic,
                "resolution_confidence": confidence,
                "confidence_tier": get_confidence_tier(confidence),
                "phase1_latency_ms": latency,
            }

    except Exception as e:
        print(f"[Phase1] Error: {e}")

    # Fallback
    return {
        "standalone_question": query,
        "detected_topic": None,
        "resolution_confidence": 0.5,
        "confidence_tier": "low",
        "phase1_latency_ms": int((time.time() - start) * 1000),
    }


def query_classifier_node(state: ChatbotState, groq_client) -> dict:
    """
    Classify query complexity for routing decisions.
    """
    query = state.get("standalone_question") or state["raw_query"]

    # Fast path patterns
    query_lower = query.lower()

    # Date/time - use injected context
    datetime_patterns = ["what day", "what date", "what time", "current date", "current time"]
    if any(p in query_lower for p in datetime_patterns):
        return {"query_complexity": "simple", "route": "datetime", "route_reason": "datetime pattern"}

    # Memory questions
    memory_patterns = ["first question", "last question", "did i ask", "our conversation"]
    if any(p in query_lower for p in memory_patterns):
        return {"query_complexity": "simple", "route": "memory", "route_reason": "memory pattern"}

    # Very short (greetings)
    if len(query.split()) <= 2:
        return {"query_complexity": "simple", "route": "direct", "route_reason": "short query"}

    # LLM classification for complex queries
    prompt = f"""Does this query need REAL-TIME web data, or is general knowledge enough?

Query: "{query}"

SEARCH needed for: current events, sports scores, weather, stock prices, recent news, current standings
KNOWLEDGE sufficient for: definitions, explanations, historical facts, how-to questions, general concepts

Reply with ONLY: SEARCH or KNOWLEDGE"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content.strip().upper()

        if "SEARCH" in result:
            return {"query_complexity": "complex", "route": "search", "route_reason": "needs real-time data"}
        else:
            return {"query_complexity": "context_dependent", "route": "direct", "route_reason": "general knowledge"}

    except Exception as e:
        print(f"[Classifier] Error: {e}")
        return {"query_complexity": "complex", "route": "search", "route_reason": "classification failed, defaulting to search"}


def route_decision_node(state: ChatbotState) -> dict:
    """
    Final routing decision based on confidence and classification.
    """
    confidence_tier = state.get("confidence_tier", "standalone")
    current_route = state.get("route", "direct")

    # Override to clarify if confidence is too low for search
    if confidence_tier == "low" and current_route == "search":
        query = state.get("standalone_question") or state["raw_query"]
        return {
            "route": "clarify",
            "route_reason": "low confidence for search query",
            "awaiting_clarification": True,
            "clarification_question": f"I want to make sure I search for the right thing. Could you clarify what you're asking about?",
        }

    return {}  # Keep existing route


def web_search_node(state: ChatbotState, perplexity_client) -> dict:
    """
    Execute web search using Perplexity.
    """
    start = time.time()
    query = state.get("standalone_question") or state["raw_query"]

    if not perplexity_client:
        return {
            "search_results": "",
            "search_latency_ms": 0,
            "search_query": query,
        }

    try:
        response = perplexity_client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": "You are a research assistant. Provide factual, well-sourced answers."},
                {"role": "user", "content": query}
            ],
            max_tokens=1000,
            temperature=0.2,
        )

        content = response.choices[0].message.content
        latency = int((time.time() - start) * 1000)

        print(f"[Search] '{query[:50]}...' in {latency}ms")

        return {
            "search_results": content,
            "search_latency_ms": latency,
            "search_query": query,
        }

    except Exception as e:
        print(f"[Search] Error: {e}")
        return {
            "search_results": "",
            "search_latency_ms": int((time.time() - start) * 1000),
            "search_query": query,
        }


def synthesis_node(state: ChatbotState, cerebras_client, system_prompt_fn: Callable[[], str]) -> dict:
    """
    Generate final response using Cerebras LLM.
    """
    start = time.time()
    query = state.get("standalone_question") or state["raw_query"]
    search_results = state.get("search_results", "")

    # Build messages
    system_prompt = system_prompt_fn()
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for turn in state.get("short_term_memory", [])[-3:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    # Add current query with search context if available
    if search_results:
        user_content = f"[Web Search Results]\n{search_results}\n\n[User Question]\n{query}"
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})

    try:
        response = cerebras_client.chat.completions.create(
            model="qwen-3-32b",
            messages=messages,
            max_tokens=150,
            temperature=0.3 if search_results else 0.5,
        )

        reply = response.choices[0].message.content.strip()

        # Strip thinking tags
        reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
        reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL)
        reply = reply.strip()

        latency = int((time.time() - start) * 1000)

        return {
            "response": reply,
            "response_latency_ms": latency,
        }

    except Exception as e:
        print(f"[Synthesis] Error: {e}")
        return {
            "response": "I'm sorry, I encountered an error processing your request.",
            "response_latency_ms": int((time.time() - start) * 1000),
        }


def clarification_node(state: ChatbotState) -> dict:
    """
    Generate clarification request.
    """
    question = state.get("clarification_question", "Could you please clarify your question?")

    return {
        "response": question,
        "awaiting_clarification": True,
    }


def memory_update_node(state: ChatbotState) -> dict:
    """
    Update conversation memory after response.
    """
    if state.get("awaiting_clarification"):
        return {}  # Don't add clarification requests to memory

    user_msg = state["raw_query"]
    assistant_msg = state.get("response", "")
    topic = state.get("detected_topic")

    # Update short-term memory
    new_memory = add_turn_to_memory(state, user_msg, assistant_msg, max_turns=5)

    # Update topic cache if we detected a topic
    new_topic_cache = state.get("topic_cache", {})
    if topic:
        new_topic_cache = update_topic_cache(state, topic)

    # Calculate total latency
    total_latency = int((time.time() - state.get("start_time", time.time())) * 1000)

    return {
        "short_term_memory": new_memory,
        "topic_cache": new_topic_cache,
        "total_latency_ms": total_latency,
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_resolve_context(state: ChatbotState) -> str:
    """Decide if we need LLM-based context resolution."""
    has_memory = len(state.get("short_term_memory", [])) > 0
    has_markers = state.get("has_context_markers", False)

    if has_memory or has_markers:
        return "context_resolution"
    else:
        return "query_classifier"


def route_after_decision(state: ChatbotState) -> str:
    """Route to appropriate execution node."""
    route = state.get("route", "direct")

    if route == "search":
        return "web_search"
    elif route == "clarify":
        return "clarification"
    else:
        return "synthesis"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def create_chatbot_graph(
    groq_client,
    perplexity_client,
    cerebras_client,
    system_prompt_fn: Callable[[], str] = None,
):
    """
    Create the LangGraph workflow for the voice assistant.

    Args:
        groq_client: Groq client for fast LLM calls (context resolution, classification)
        perplexity_client: Perplexity client for web search
        cerebras_client: Cerebras client for synthesis
        system_prompt_fn: Function that returns current system prompt (with date/time)

    Returns:
        Compiled LangGraph workflow
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not installed. Run: pip install langgraph")

    # Default system prompt
    if system_prompt_fn is None:
        system_prompt_fn = lambda: "You are a helpful voice assistant. Be concise."

    # Create graph
    workflow = StateGraph(ChatbotState)

    # Add nodes with bound clients
    workflow.add_node("context_detection", context_detection_node)
    workflow.add_node("context_resolution", partial(context_resolution_node, groq_client=groq_client))
    workflow.add_node("query_classifier", partial(query_classifier_node, groq_client=groq_client))
    workflow.add_node("route_decision", route_decision_node)
    workflow.add_node("web_search", partial(web_search_node, perplexity_client=perplexity_client))
    workflow.add_node("synthesis", partial(synthesis_node, cerebras_client=cerebras_client, system_prompt_fn=system_prompt_fn))
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("memory_update", memory_update_node)

    # Entry point
    workflow.set_entry_point("context_detection")

    # Edges
    workflow.add_conditional_edges(
        "context_detection",
        should_resolve_context,
        {
            "context_resolution": "context_resolution",
            "query_classifier": "query_classifier",
        }
    )

    workflow.add_edge("context_resolution", "query_classifier")
    workflow.add_edge("query_classifier", "route_decision")

    workflow.add_conditional_edges(
        "route_decision",
        route_after_decision,
        {
            "web_search": "web_search",
            "clarification": "clarification",
            "synthesis": "synthesis",
        }
    )

    workflow.add_edge("web_search", "synthesis")
    workflow.add_edge("synthesis", "memory_update")
    workflow.add_edge("clarification", "memory_update")
    workflow.add_edge("memory_update", END)

    # Compile with memory checkpointer
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_query(
    graph,
    query: str,
    session_id: str = "default",
    short_term_memory: list = None,
    topic_cache: dict = None,
) -> ChatbotState:
    """
    Run a query through the chatbot graph.

    Args:
        graph: Compiled LangGraph workflow
        query: User's query
        session_id: Session identifier for checkpointing
        short_term_memory: Previous conversation turns
        topic_cache: Active topics

    Returns:
        Final ChatbotState with response
    """
    initial_state = create_initial_state(
        raw_query=query,
        session_id=session_id,
        short_term_memory=short_term_memory,
        topic_cache=topic_cache,
    )

    config = {"configurable": {"thread_id": session_id}}

    result = await graph.ainvoke(initial_state, config)
    return result


def run_query_sync(
    graph,
    query: str,
    session_id: str = "default",
    short_term_memory: list = None,
    topic_cache: dict = None,
) -> ChatbotState:
    """
    Synchronous version of run_query.
    """
    initial_state = create_initial_state(
        raw_query=query,
        session_id=session_id,
        short_term_memory=short_term_memory,
        topic_cache=topic_cache,
    )

    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke(initial_state, config)
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "create_chatbot_graph",
    "run_query",
    "run_query_sync",
    "LANGGRAPH_AVAILABLE",
]
