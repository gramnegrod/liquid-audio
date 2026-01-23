"""
Memory Service for Voice Assistant

Provides persistent storage for conversation state using:
- Redis: Short-term memory, topic cache (TTL: 1 hour per session)
- PostgreSQL: Long-term memory, user preferences (future)

Based on: "Generative Agents" (Park et al., 2023) memory stream architecture

Usage:
    from memory_service import MemoryService

    memory = MemoryService()  # Uses in-memory fallback if Redis unavailable

    # Save conversation state
    await memory.save_session(session_id, short_term_memory, topic_cache)

    # Load conversation state
    state = await memory.load_session(session_id)
"""

import json
import time
import asyncio
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import os

# Redis imports (optional)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[Memory] Redis not installed. Using in-memory fallback.")

from chatbot_state import ConversationTurn, TopicEntity


# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
SESSION_TTL = 3600  # 1 hour
TOPIC_TTL = 300     # 5 minutes for topics
MAX_MEMORY_TURNS = 10  # Max turns to store per session


@dataclass
class SessionState:
    """Complete session state for persistence."""
    session_id: str
    short_term_memory: List[ConversationTurn]
    topic_cache: Dict[str, TopicEntity]
    last_updated: float
    turn_count: int


# ============================================================================
# MEMORY SERVICE
# ============================================================================

class MemoryService:
    """
    Persistent memory service with Redis backend.

    Falls back to in-memory storage if Redis is unavailable.
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or REDIS_URL
        self._redis: Optional[redis.Redis] = None
        self._fallback_store: Dict[str, SessionState] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        if not REDIS_AVAILABLE:
            print("[Memory] Redis not available, using in-memory fallback")
            return False

        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            self._connected = True
            print(f"[Memory] Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            print(f"[Memory] Redis connection failed: {e}, using in-memory fallback")
            self._redis = None
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"voice_assistant:session:{session_id}"

    def _topic_key(self, session_id: str) -> str:
        """Generate Redis key for topic cache."""
        return f"voice_assistant:topics:{session_id}"

    # ========================================================================
    # SESSION OPERATIONS
    # ========================================================================

    async def save_session(
        self,
        session_id: str,
        short_term_memory: List[ConversationTurn],
        topic_cache: Dict[str, TopicEntity],
    ) -> bool:
        """
        Save session state to storage.

        Args:
            session_id: Unique session identifier
            short_term_memory: List of conversation turns
            topic_cache: Active topic entities

        Returns:
            True if save successful
        """
        state = SessionState(
            session_id=session_id,
            short_term_memory=short_term_memory[-MAX_MEMORY_TURNS:],
            topic_cache=topic_cache,
            last_updated=time.time(),
            turn_count=len(short_term_memory),
        )

        if self._connected and self._redis:
            return await self._save_to_redis(state)
        else:
            return self._save_to_memory(state)

    async def _save_to_redis(self, state: SessionState) -> bool:
        """Save session to Redis."""
        try:
            key = self._session_key(state.session_id)
            data = json.dumps(asdict(state))
            await self._redis.setex(key, SESSION_TTL, data)

            # Also save topics with shorter TTL
            if state.topic_cache:
                topic_key = self._topic_key(state.session_id)
                topic_data = json.dumps(state.topic_cache)
                await self._redis.setex(topic_key, TOPIC_TTL, topic_data)

            return True
        except Exception as e:
            print(f"[Memory] Redis save failed: {e}")
            return self._save_to_memory(state)

    def _save_to_memory(self, state: SessionState) -> bool:
        """Save session to in-memory fallback."""
        self._fallback_store[state.session_id] = state
        return True

    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load session state from storage.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        if self._connected and self._redis:
            state = await self._load_from_redis(session_id)
            if state:
                return state

        return self._load_from_memory(session_id)

    async def _load_from_redis(self, session_id: str) -> Optional[SessionState]:
        """Load session from Redis."""
        try:
            key = self._session_key(session_id)
            data = await self._redis.get(key)

            if data:
                parsed = json.loads(data)
                return SessionState(**parsed)
            return None
        except Exception as e:
            print(f"[Memory] Redis load failed: {e}")
            return None

    def _load_from_memory(self, session_id: str) -> Optional[SessionState]:
        """Load session from in-memory fallback."""
        return self._fallback_store.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete session from storage."""
        if self._connected and self._redis:
            try:
                key = self._session_key(session_id)
                topic_key = self._topic_key(session_id)
                await self._redis.delete(key, topic_key)
            except Exception as e:
                print(f"[Memory] Redis delete failed: {e}")

        if session_id in self._fallback_store:
            del self._fallback_store[session_id]

        return True

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    async def get_memory(self, session_id: str) -> List[ConversationTurn]:
        """Get short-term memory for session."""
        state = await self.load_session(session_id)
        return state.short_term_memory if state else []

    async def get_topics(self, session_id: str) -> Dict[str, TopicEntity]:
        """Get topic cache for session."""
        state = await self.load_session(session_id)
        return state.topic_cache if state else {}

    async def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        topic: Optional[str] = None,
        search_used: bool = False,
    ) -> bool:
        """
        Add a conversation turn to the session.

        Convenience method that handles load, update, save.
        """
        state = await self.load_session(session_id)

        if state:
            memory = state.short_term_memory
            topics = state.topic_cache
        else:
            memory = []
            topics = {}

        # Add new turn
        new_turn: ConversationTurn = {
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": time.time(),
            "topic": topic,
            "search_used": search_used,
        }
        memory.append(new_turn)

        # Update topic cache if provided
        if topic:
            now = time.time()
            if topic in topics:
                topics[topic]["last_referenced"] = now
                topics[topic]["mention_count"] += 1
            else:
                topics[topic] = TopicEntity(
                    name=topic,
                    entity_type="general",
                    first_mentioned=now,
                    last_referenced=now,
                    mention_count=1,
                )

        # Save updated state
        return await self.save_session(session_id, memory, topics)

    async def clear_session(self, session_id: str) -> bool:
        """Clear all memory for a session."""
        return await self.delete_session(session_id)

    # ========================================================================
    # STATS & DEBUGGING
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory service statistics."""
        stats = {
            "connected": self._connected,
            "backend": "redis" if self._connected else "in-memory",
            "fallback_sessions": len(self._fallback_store),
        }

        if self._connected and self._redis:
            try:
                info = await self._redis.info("keyspace")
                stats["redis_info"] = info
            except Exception:
                pass

        return stats


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_memory_service: Optional[MemoryService] = None


async def get_memory_service() -> MemoryService:
    """Get or create the memory service singleton."""
    global _memory_service

    if _memory_service is None:
        _memory_service = MemoryService()
        await _memory_service.connect()

    return _memory_service


def get_memory_service_sync() -> MemoryService:
    """Synchronous version - creates service without connecting."""
    global _memory_service

    if _memory_service is None:
        _memory_service = MemoryService()

    return _memory_service


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MemoryService",
    "SessionState",
    "get_memory_service",
    "get_memory_service_sync",
    "REDIS_AVAILABLE",
]
