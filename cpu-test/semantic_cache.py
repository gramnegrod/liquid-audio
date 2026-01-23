"""
Semantic Cache for Voice Assistant

Caches responses based on semantic similarity to reduce latency and API costs.
Uses embedding-based similarity with configurable threshold.

Architecture:
- Embeddings: bge-small-en-v1.5 (local, fast)
- Vector Store: ChromaDB (local, persistent)
- Similarity Threshold: 0.8 (industry standard)

Based on: GPT-5.2 Architect recommendations

Usage:
    from semantic_cache import SemanticCache

    cache = SemanticCache()

    # Check cache before API call
    cached = await cache.get(query)
    if cached:
        return cached

    # After getting response, cache it
    await cache.set(query, response, metadata={"route": "search"})
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import os

# ChromaDB imports (optional)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[Cache] ChromaDB not installed. Run: pip install chromadb")

# Sentence Transformers for embeddings (optional)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("[Cache] sentence-transformers not installed. Run: pip install sentence-transformers")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Embedding model - bge-small is fast and good quality
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Similarity threshold - queries with similarity > 0.8 are considered cache hits
SIMILARITY_THRESHOLD = 0.8

# Cache TTL in seconds (1 hour default)
CACHE_TTL = 3600

# Maximum cache size per collection
MAX_CACHE_SIZE = 10000


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    query: str
    response: str
    route: str  # search, direct, etc.
    timestamp: float
    latency_ms: int
    confidence: float


# ============================================================================
# SEMANTIC CACHE
# ============================================================================

class SemanticCache:
    """
    Semantic cache using embeddings and vector similarity.

    Falls back to exact-match caching if ChromaDB/embeddings unavailable.
    """

    def __init__(
        self,
        collection_name: str = "voice_assistant_cache",
        persist_directory: str = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(__file__), ".cache", "chroma"
        )

        self._client: Optional[chromadb.Client] = None
        self._collection = None
        self._embedder: Optional[SentenceTransformer] = None
        self._fallback_cache: Dict[str, CacheEntry] = {}
        self._initialized = False

        # Stats
        self._hits = 0
        self._misses = 0

    def initialize(self) -> bool:
        """Initialize ChromaDB and embeddings. Returns True if successful."""
        if self._initialized:
            return True

        # Initialize embeddings
        if EMBEDDINGS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(EMBEDDING_MODEL)
                print(f"[Cache] Loaded embedding model: {EMBEDDING_MODEL}")
            except Exception as e:
                print(f"[Cache] Failed to load embeddings: {e}")
                self._embedder = None

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE and self._embedder:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                self._initialized = True
                print(f"[Cache] ChromaDB initialized at {self.persist_directory}")
                return True
            except Exception as e:
                print(f"[Cache] ChromaDB init failed: {e}")

        print("[Cache] Using in-memory fallback cache")
        return False

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        if self._embedder is None:
            return None

        try:
            embedding = self._embedder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"[Cache] Embedding error: {e}")
            return None

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact-match fallback."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    # ========================================================================
    # CACHE OPERATIONS
    # ========================================================================

    def get(self, query: str) -> Optional[CacheEntry]:
        """
        Get cached response for query.

        Uses semantic similarity if embeddings available, else exact match.

        Args:
            query: User query to look up

        Returns:
            CacheEntry if hit, None if miss
        """
        if not self._initialized:
            self.initialize()

        # Try semantic search
        if self._collection and self._embedder:
            result = self._semantic_search(query)
            if result:
                self._hits += 1
                return result

        # Fall back to exact match
        query_hash = self._hash_query(query)
        if query_hash in self._fallback_cache:
            entry = self._fallback_cache[query_hash]
            # Check TTL
            if time.time() - entry.timestamp < CACHE_TTL:
                self._hits += 1
                return entry
            else:
                del self._fallback_cache[query_hash]

        self._misses += 1
        return None

    def _semantic_search(self, query: str) -> Optional[CacheEntry]:
        """Search cache using semantic similarity."""
        embedding = self._get_embedding(query)
        if not embedding:
            return None

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )

            if not results["ids"][0]:
                return None

            # Check similarity threshold (distance is 1 - similarity for cosine)
            distance = results["distances"][0][0]
            similarity = 1 - distance

            if similarity < self.similarity_threshold:
                return None

            # Check TTL
            metadata = results["metadatas"][0][0]
            timestamp = metadata.get("timestamp", 0)
            if time.time() - timestamp > CACHE_TTL:
                # Expired - delete and return miss
                self._collection.delete(ids=[results["ids"][0][0]])
                return None

            print(f"[Cache] HIT (similarity={similarity:.3f}): '{query[:40]}...'")

            return CacheEntry(
                query=results["documents"][0][0],
                response=metadata.get("response", ""),
                route=metadata.get("route", "unknown"),
                timestamp=timestamp,
                latency_ms=metadata.get("latency_ms", 0),
                confidence=similarity,
            )

        except Exception as e:
            print(f"[Cache] Search error: {e}")
            return None

    def set(
        self,
        query: str,
        response: str,
        route: str = "unknown",
        latency_ms: int = 0,
    ) -> bool:
        """
        Cache a response.

        Args:
            query: User query
            response: Assistant response
            route: Route taken (search, direct, etc.)
            latency_ms: Response latency

        Returns:
            True if cached successfully
        """
        if not self._initialized:
            self.initialize()

        entry = CacheEntry(
            query=query,
            response=response,
            route=route,
            timestamp=time.time(),
            latency_ms=latency_ms,
            confidence=1.0,
        )

        # Try semantic cache
        if self._collection and self._embedder:
            success = self._semantic_set(entry)
            if success:
                return True

        # Fall back to exact match
        query_hash = self._hash_query(query)
        self._fallback_cache[query_hash] = entry

        # Prune fallback cache if too large
        if len(self._fallback_cache) > MAX_CACHE_SIZE:
            self._prune_fallback_cache()

        return True

    def _semantic_set(self, entry: CacheEntry) -> bool:
        """Add entry to semantic cache."""
        embedding = self._get_embedding(entry.query)
        if not embedding:
            return False

        try:
            # Use query hash as ID
            doc_id = self._hash_query(entry.query)

            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[entry.query],
                metadatas=[{
                    "response": entry.response,
                    "route": entry.route,
                    "timestamp": entry.timestamp,
                    "latency_ms": entry.latency_ms,
                }]
            )
            return True

        except Exception as e:
            print(f"[Cache] Set error: {e}")
            return False

    def _prune_fallback_cache(self):
        """Remove oldest entries from fallback cache."""
        if len(self._fallback_cache) <= MAX_CACHE_SIZE // 2:
            return

        # Sort by timestamp and keep newest half
        sorted_items = sorted(
            self._fallback_cache.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        self._fallback_cache = dict(sorted_items[:MAX_CACHE_SIZE // 2])

    def invalidate(self, query: str) -> bool:
        """Remove a specific entry from cache."""
        query_hash = self._hash_query(query)

        # Remove from fallback
        if query_hash in self._fallback_cache:
            del self._fallback_cache[query_hash]

        # Remove from ChromaDB
        if self._collection:
            try:
                self._collection.delete(ids=[query_hash])
            except Exception:
                pass

        return True

    def clear(self) -> bool:
        """Clear all cached entries."""
        self._fallback_cache.clear()

        if self._collection:
            try:
                # Delete and recreate collection
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[Cache] Clear error: {e}")

        self._hits = 0
        self._misses = 0
        return True

    # ========================================================================
    # STATS
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        stats = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "fallback_size": len(self._fallback_cache),
            "backend": "chromadb" if self._collection else "in-memory",
            "threshold": self.similarity_threshold,
        }

        if self._collection:
            try:
                stats["collection_size"] = self._collection.count()
            except Exception:
                pass

        return stats


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_cache_instance: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the semantic cache singleton."""
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = SemanticCache()
        _cache_instance.initialize()

    return _cache_instance


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SemanticCache",
    "CacheEntry",
    "get_semantic_cache",
    "CHROMADB_AVAILABLE",
    "EMBEDDINGS_AVAILABLE",
    "SIMILARITY_THRESHOLD",
]
