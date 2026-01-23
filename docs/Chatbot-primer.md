# Chatbot Development Primer
## Research-Backed Patterns for General Conversational AI

*Last Updated: January 2026*
*Based on: ACL, EMNLP, TACL papers and industry best practices*

---

## Table of Contents

1. [Context Resolution](#1-context-resolution)
2. [Coreference & Ellipsis](#2-coreference--ellipsis)
3. [Memory Architecture](#3-memory-architecture)
4. [Reflection Patterns](#4-reflection-patterns)
5. [Confidence Calibration](#5-confidence-calibration)
6. [Routing for Latency](#6-routing-for-latency)
7. [Implementation Patterns](#7-implementation-patterns)
8. [Anti-Patterns](#8-anti-patterns)
9. [References](#9-references)

**Addendum:** [Reflections on Modern AI Chatbot Design](#addendum-reflections-on-modern-ai-chatbot-design)

---

## 1. Context Resolution

### The Problem

In multi-turn conversations, users naturally use language shortcuts:
- **Pronouns:** "What about *them*?" "How does *it* work?"
- **Demonstratives:** "*That* sounds good" "*This* is confusing"
- **Ellipsis:** "And the price?" (missing subject entirely)

A general chatbot must resolve these references WITHOUT domain-specific knowledge.

### The Solution: Standalone Question Reformulation

**Industry Standard Pattern (LangChain/LlamaIndex):**

```
Given a chat history and the latest user question which might reference
context in the chat history, formulate a standalone question which can
be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.
```

**Origin:** This pattern comes from the TREC Conversational Assistance Track (CAsT) and the CANARD dataset research.

### Key Research

**Foundational Paper:**
> **"Can You Unpack That? Learning to Rewrite Questions-in-Context"**
> Ahmed Elgohary, Denis Peskov, Jordan Boyd-Graber
> EMNLP 2019
> https://aclanthology.org/D19-1605/

- Introduced **CANARD dataset**: 40,527 question rewrites
- Specifically targets **coreference and ellipsis resolution**
- Questions rewritten to be self-contained using ONLY linguistic patterns

**Why It Works:**
The reformulation is purely linguistic - detecting pronouns, demonstratives, and missing elements - NOT matching domain keywords like "sports" or "finance."

---

## 2. Coreference & Ellipsis

### The 70% Rule

**Key Paper:**
> **"CoQA: A Conversational Question Answering Challenge"**
> Siva Reddy, Danqi Chen, Christopher D. Manning
> TACL 2019
> https://aclanthology.org/Q19-1016/

This paper quantified context-dependence in conversational questions:

| Question Type | Percentage | Detection Method |
|---------------|------------|------------------|
| Explicit coreference (pronouns) | **49.7%** | Pronoun detection |
| Ellipsis (implicit reference) | **19.8%** | Missing subject/object |
| Standalone | 30.5% | No markers present |

**Implication:** ~70% of conversational questions require context resolution, and they're detectable through **linguistic markers alone**.

### Linguistic Markers Taxonomy

**From Jurafsky & Martin, Speech and Language Processing, Chapter 26:**
> https://web.stanford.edu/~jurafsky/slp3/26.pdf

Four types of referring expressions:

1. **Pronouns:** it, they, them, he, she, we, its, their
2. **Demonstratives:** this, that, these, those
3. **Definite NPs:** "the results", "the answer", "the problem"
4. **Zero anaphora (ellipsis):** Missing subject/object entirely

**Spoken Dialogue Statistics:**
- 51% of references in dialogue are pronouns
- Only 18.4% in written text
- Dialogue is MORE pronoun-heavy, making linguistic detection more effective

### Detection Heuristic

```python
CONTEXT_MARKERS = [
    # Pronouns
    'it', 'they', 'them', 'he', 'she', 'we', 'its', 'their', 'him', 'her',
    # Demonstratives
    'this', 'that', 'these', 'those',
    # Continuations
    'the same', 'another', 'more', 'also', 'again', 'too',
    # Implicit references
    'the results', 'the answer', 'the problem', 'the issue'
]

def likely_needs_context(query: str) -> bool:
    """Fast heuristic before calling LLM for reformulation."""
    query_lower = query.lower()
    tokens = query_lower.split()
    return any(marker in tokens or marker in query_lower
               for marker in CONTEXT_MARKERS)
```

---

## 3. Memory Architecture

### The Memory Stream Pattern

**Landmark Paper:**
> **"Generative Agents: Interactive Simulacra of Human Behavior"**
> Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, et al.
> Stanford University & Google Research
> UIST 2023
> https://arxiv.org/abs/2304.03442

**Architecture Components:**

```
┌─────────────────────────────────────────────────┐
│                 MEMORY STREAM                    │
│  (Chronological log of all observations)        │
├─────────────────────────────────────────────────┤
│                   RETRIEVAL                      │
│  (Recency × Importance × Relevance scoring)     │
├─────────────────────────────────────────────────┤
│                  REFLECTION                      │
│  (Higher-level synthesis of memories)           │
└─────────────────────────────────────────────────┘
```

**Key Insight:**
> "Reflections are stored in the memory stream, including pointers to the memory objects that were cited"

Ablation study showed **reflection is critical** to coherent long-term behavior.

### Memory Types

**From "A Survey on the Memory Mechanism of Large Language Model-based Agents"**
> ACM Transactions on Intelligent Systems and Technology, 2024
> https://dl.acm.org/doi/10.1145/3748302

| Memory Type | Purpose | Example |
|-------------|---------|---------|
| **Sensory** | Raw input buffer | Current user message |
| **Short-term** | Active context | Last 3-5 turns |
| **Long-term** | Persistent storage | Topic cache, user preferences |
| **Episodic** | Event sequences | "Last time we discussed X..." |
| **Semantic** | Factual knowledge | "User prefers formal tone" |

### Practical Implementation

```python
class ConversationMemory:
    def __init__(self, max_short_term=5):
        self.short_term = []  # Recent turns
        self.topic_cache = {} # Current topic entity
        self.long_term = {}   # Persistent facts

    def add_turn(self, user_msg, assistant_msg, topic=None):
        self.short_term.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": time.time()
        })
        if len(self.short_term) > self.max_short_term:
            self._consolidate_to_long_term()
        if topic:
            self.topic_cache["current"] = topic

    def get_context_for_resolution(self) -> str:
        """Get context string for standalone question reformulation."""
        context_parts = []
        if self.topic_cache.get("current"):
            context_parts.append(f"Current topic: {self.topic_cache['current']}")
        for turn in self.short_term[-3:]:  # Last 3 turns
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant'][:200]}...")
        return "\n".join(context_parts)
```

---

## 4. Reflection Patterns

### What is Reflection?

Reflection is when an agent **synthesizes higher-level insights** from raw memories before responding. This prevents context pollution and improves coherence.

**Key Paper:**
> **"Reflexion: Language Agents with Verbal Reinforcement Learning"**
> Noah Shinn, Federico Cassano, et al.
> 2023
> https://arxiv.org/abs/2303.11366

**Results:** Achieved 91% on HumanEval vs GPT-4's 80% by using episodic memory reflection.

### Reflection Before Answering

Instead of dumping entire conversation history into context:

```
❌ BAD: Pass 20 turns of raw conversation to LLM

✅ GOOD:
1. Retrieve relevant memories
2. Reflect: "What do I know about this topic from our conversation?"
3. Generate concise summary
4. Use summary (not raw history) for response
```

### Implementation Pattern

```python
async def reflect_on_context(self, query: str, memories: list) -> str:
    """Generate reflection before answering."""
    reflection_prompt = f"""
    Based on these conversation memories, what is relevant to answering: "{query}"?

    Memories:
    {self._format_memories(memories)}

    Provide a 2-3 sentence summary of relevant context only.
    """

    reflection = await self.llm.generate(reflection_prompt)
    return reflection
```

### Long-Term Conversation Benchmarks

**LoCoMo Benchmark:**
> **"Evaluating Very Long-Term Conversational Memory of LLM Agents"**
> ACL 2024
> https://aclanthology.org/2024.acl-long.747.pdf

- Tests conversations spanning **300 turns, 9K tokens**
- Evaluates temporal reasoning, factual consistency, multi-session memory
- Current LLMs struggle without explicit memory architecture

---

## 5. Confidence Calibration

### The Calibration Problem

Modern neural networks (including LLMs) are often **poorly calibrated** - their confidence scores don't match actual accuracy.

**Foundational Paper:**
> **"On Calibration of Modern Neural Networks"**
> Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
> ICML 2017
> https://arxiv.org/abs/1706.04599

Introduced **temperature scaling** for calibration.

### Verbalized Confidence is Better

**Key Paper:**
> **"Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models"**
> Katherine Tian, Eric Mitchell, et al.
> Stanford University
> EMNLP 2023
> https://aclanthology.org/2023.emnlp-main.330/

**Finding:** For RLHF models (ChatGPT, GPT-4, Claude):
> "Verbalized confidence is better-calibrated than token probabilities, often reducing expected calibration error by a relative 50%"

**Implication:** Ask the LLM to output a confidence score as text, don't rely on logprobs.

### The 70/90 Threshold Model

**Industry Practice (Klarna Case Study):**
> https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/

- Klarna's AI handled **2/3 of customer service chats**
- Key strategy: "If AI wasn't certain about an answer, it would escalate to human"

**Recommended Thresholds:**

| Confidence | Action | Use Case |
|------------|--------|----------|
| **90-100%** | Proceed autonomously | Clear reformulation, high certainty |
| **70-89%** | Proceed with caution | Probable match, monitor for errors |
| **40-69%** | Request clarification | Ambiguous, ask user to clarify |
| **0-39%** | Treat as new topic | No context match, standalone query |

**Containment Rate Benchmarks:**
> https://gettalkative.com/info/chatbot-containment-rate

- Well-designed chatbots: **70-80% containment rate**
- Enterprise target: **85-90% for routine queries**
- Complex issues: Always escalate to human

### Implementation

```python
async def resolve_with_confidence(self, query: str, context: str) -> dict:
    prompt = f"""
    ...reformulation prompt...

    CONFIDENCE CALIBRATION:
    - 0.90-1.00: Clear linguistic marker present, reformulation certain
    - 0.70-0.89: Probable reference to chat history
    - 0.40-0.69: Ambiguous, could be standalone
    - 0.00-0.39: Standalone question or completely new topic

    OUTPUT JSON: {{"standalone_question": "...", "topic": "...", "confidence": 0.0-1.0}}
    """

    result = await self.llm.generate(prompt)

    # Route based on confidence
    if result["confidence"] >= 0.90:
        return self.proceed_with_reformulation(result)
    elif result["confidence"] >= 0.70:
        return self.proceed_with_logging(result)  # Monitor for errors
    elif result["confidence"] >= 0.40:
        return self.request_clarification(query)
    else:
        return self.treat_as_standalone(query)
```

---

## 6. Routing for Latency

### The Routing Problem

Not all queries need the same processing:
- "Hi" → Simple response, no search needed
- "What's 2+2?" → Direct answer, no context needed
- "Tell me more about that" → Needs context resolution
- "Explain quantum entanglement in detail" → Needs search + synthesis

### Complexity-Based Routing

**ModernBERT for Classification:**
Small, fast classifier to route queries before expensive LLM calls.

```
Query → [Classifier] → Route
                ↓
    ┌───────────────────────────────────────────────────┐
    │ SIMPLE (greeting, math, factoid)  │ → Direct response
    │ CONTEXT-DEPENDENT (pronouns)      │ → Context resolution → Response
    │ COMPLEX (explanation, research)   │ → Search → Synthesis → Response
    └───────────────────────────────────────────────────┘
```

### Semantic Caching

**Pattern:** Cache responses to semantically similar queries.

**Threshold:** Cosine similarity > 0.8 (industry standard)

```python
class SemanticCache:
    def __init__(self, threshold=0.8):
        self.cache = {}  # embedding -> response
        self.threshold = threshold

    async def get_or_compute(self, query: str, compute_fn):
        query_embedding = await self.embed(query)

        for cached_embedding, response in self.cache.items():
            similarity = cosine_similarity(query_embedding, cached_embedding)
            if similarity > self.threshold:
                return response  # Cache hit

        # Cache miss - compute and store
        response = await compute_fn(query)
        self.cache[query_embedding] = response
        return response
```

### Latency Targets

| Route | Target Latency | Method |
|-------|----------------|--------|
| Simple/Cached | < 100ms | Direct response or cache hit |
| Context Resolution | < 200ms | Fast LLM (Groq Llama 3.1 8B) |
| Standard Query | < 1s | Primary LLM |
| Complex/Search | < 3s | Search + Synthesis |

---

## 7. Implementation Patterns

### The Two-Phase Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 1: CONTEXT RESOLUTION              │
│                                                              │
│  Input: Raw query + Conversation memory                      │
│  Output: Standalone question + Topic + Confidence            │
│                                                              │
│  Method: LangChain-style reformulation prompt                │
│  Model: Fast LLM (Groq Llama 3.1 8B, ~100-150ms)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 2: ROUTING                         │
│                                                              │
│  Input: Standalone question + Confidence                     │
│  Output: Route decision (simple/search/complex)              │
│                                                              │
│  Method: Confidence thresholds + Query classification        │
│  Fast path: High confidence → direct to response generation  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     PHASE 3: EXECUTION                       │
│                                                              │
│  Simple → Direct LLM response                                │
│  Search → RAG pipeline → Synthesized response                │
│  Complex → Multi-step reasoning → Comprehensive response     │
└─────────────────────────────────────────────────────────────┘
```

### The Reformulation Prompt (Domain-Agnostic)

```python
REFORMULATION_PROMPT = """
Given a chat history and the latest user question which might reference
context in the chat history, formulate a standalone question which can
be understood without the chat history.

CHAT HISTORY:
{context}

LATEST QUESTION: "{query}"

INSTRUCTIONS:
- If the question contains references that depend on chat history,
  rewrite it to be self-contained
- If the question is already standalone or introduces a new topic,
  return it unchanged
- Do NOT answer the question - only reformulate if needed
- Extract the current topic being discussed (null if new/unrelated)

REFORMULATION TRIGGERS (linguistic markers):
- Pronouns: it, they, them, he, she, we, its, their
- Demonstratives: this, that, these, those
- Definite articles with implicit reference: "the results", "the answer"
- Continuations: the same, another, more, also, again, next, previous
- Ellipsis: questions missing subject/object that chat history provides

CONFIDENCE CALIBRATION:
- 0.90-1.00: Clear linguistic marker present, reformulation certain
- 0.70-0.89: Probable reference to chat history
- 0.40-0.69: Ambiguous, could be standalone
- 0.00-0.39: Standalone question or completely new topic

OUTPUT JSON ONLY:
{{"standalone_question": "...", "topic": "..." or null, "confidence": 0.0-1.0}}
"""
```

---

## 8. Anti-Patterns

### ❌ Domain-Specific Examples in Prompts

**Bad:**
```
DECISION RULES:
- USE CONTEXT if query mentions: "the losses", "their record", "the coach"
```

**Why it fails:** Only works for sports. Doesn't generalize to finance, tech, cooking, etc.

**Good:**
```
REFORMULATION TRIGGERS:
- Pronouns: it, they, them, he, she
- Demonstratives: this, that, these, those
```

**Why it works:** Linguistic patterns are universal across ALL domains.

### ❌ Keyword Matching for Context Detection

**Bad:**
```python
if "it" in query or "that" in query:
    needs_context = True
```

**Why it fails:** "It is raining" doesn't need context. "That's interesting" might not either.

**Good:**
```python
# Use LLM judgment with linguistic guidance
prompt = "Does this question require chat history to understand?..."
```

### ❌ Dumping Full History into Context

**Bad:**
```python
context = "\n".join([str(turn) for turn in self.all_turns])  # 50 turns!
```

**Why it fails:** Context pollution, token waste, relevance dilution.

**Good:**
```python
context = self.get_relevant_context(query, max_turns=3)
# Or use reflection to summarize relevant memories
```

### ❌ Single Confidence Threshold

**Bad:**
```python
if confidence > 0.5:
    proceed()
else:
    ask_clarification()
```

**Why it fails:** No nuance. 0.51 and 0.99 treated the same.

**Good:**
```python
if confidence >= 0.90:
    proceed_confidently()
elif confidence >= 0.70:
    proceed_with_monitoring()
elif confidence >= 0.40:
    request_clarification()
else:
    treat_as_standalone()
```

### ❌ Ignoring Ellipsis

**Bad:** Only detecting pronouns.

**Why it fails:** "And the price?" has no pronouns but clearly needs context.

**Good:** Detect ellipsis patterns (questions missing expected subjects/objects).

---

## 9. References

### Foundational Papers

| Paper | Authors | Venue | Year | Key Contribution |
|-------|---------|-------|------|------------------|
| Can You Unpack That? | Elgohary et al. | EMNLP | 2019 | CANARD dataset, question rewriting |
| CoQA | Reddy, Chen, Manning | TACL | 2019 | 70% context-dependence statistic |
| Generative Agents | Park et al. | UIST | 2023 | Memory stream + reflection architecture |
| Reflexion | Shinn et al. | arXiv | 2023 | Verbal reinforcement learning |
| Just Ask for Calibration | Tian et al. | EMNLP | 2023 | Verbalized confidence calibration |
| On Calibration of Neural Networks | Guo et al. | ICML | 2017 | Temperature scaling |

### URLs

**Standalone Question Reformulation:**
- CANARD Paper: https://aclanthology.org/D19-1605/
- LlamaIndex Condense Mode: https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_question/
- TREC CAsT 2021: https://trec.nist.gov/pubs/trec30/papers/Overview-CAsT.pdf

**Coreference Resolution:**
- Jurafsky & Martin SLP3 Ch.26: https://web.stanford.edu/~jurafsky/slp3/26.pdf
- CoQA Paper: https://aclanthology.org/Q19-1016/

**Memory & Reflection:**
- Generative Agents: https://arxiv.org/abs/2304.03442
- Reflexion: https://arxiv.org/abs/2303.11366
- LoCoMo Benchmark: https://aclanthology.org/2024.acl-long.747.pdf
- Memory Survey: https://dl.acm.org/doi/10.1145/3748302

**Confidence Calibration:**
- Just Ask for Calibration: https://aclanthology.org/2023.emnlp-main.330/
- Neural Network Calibration: https://arxiv.org/abs/1706.04599
- Klarna Case Study: https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/
- Containment Benchmarks: https://gettalkative.com/info/chatbot-containment-rate

**Conversational QA Datasets:**
- QuAC: https://quac.ai/
- QReCC: https://github.com/apple/ml-qrecc
- TREC CAsT: https://www.treccast.ai/

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────┐
│                    CHATBOT QUICK REFERENCE                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  CONTEXT DETECTION (Linguistic Markers):                    │
│  • Pronouns: it, they, them, he, she, we                   │
│  • Demonstratives: this, that, these, those                │
│  • Continuations: more, also, again, another               │
│  • Ellipsis: missing subject/object                        │
│                                                             │
│  CONFIDENCE THRESHOLDS:                                     │
│  • 90%+ → Proceed autonomously                             │
│  • 70-89% → Proceed with monitoring                        │
│  • 40-69% → Request clarification                          │
│  • <40% → Treat as standalone/new topic                    │
│                                                             │
│  REFORMULATION PROMPT (LangChain pattern):                  │
│  "Formulate a standalone question which can be              │
│   understood without the chat history"                      │
│                                                             │
│  KEY STATISTIC:                                             │
│  ~70% of conversational questions need context resolution   │
│  (49.7% pronouns + 19.8% ellipsis) - CoQA 2019             │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

*Document generated from research on conversational AI best practices, January 2026.*

---
---

# Addendum: Reflections on Modern AI Chatbot Design

*A narrative synthesis of best practices for AI chatbots with memory and web ability*

---

## The Fundamental Insight

The most important lesson from the research is counterintuitive: **the key to building a general chatbot is to know nothing about any specific domain**.

When we tried to teach the system about sports ("the losses", "their record"), we created a system that only worked for sports. But when we stepped back and asked "what makes a question context-dependent?", we discovered something universal: **it's not about topics, it's about language itself**.

The CoQA research proved this empirically. Across 8,000 conversations spanning every domain imaginable, exactly 49.7% of questions used pronouns and 19.8% used ellipsis. That's not a sports pattern or a finance pattern—that's a *human language* pattern. Pronouns exist because humans are efficient communicators. We say "them" instead of repeating "the Texas Tech women's basketball team" because our brains naturally track context.

A well-designed chatbot must do the same.

---

## Memory Is Not Storage—It's Cognition

The naive approach to chatbot memory is to store everything and dump it into the prompt. This fails catastrophically. Context windows fill up. Relevance dilutes. The model drowns in its own history.

The Stanford Generative Agents research revealed something profound: **memory without reflection is just logging**. What made their agents believable wasn't that they remembered everything—it was that they *thought about* what they remembered.

The architecture has three layers:

1. **The Memory Stream** - A chronological log of everything that happened. Raw, unprocessed, complete.

2. **Retrieval** - When a new query arrives, the system doesn't dump the whole stream. It asks: "What memories are *relevant* to this moment?" Relevance is scored by recency (recent memories matter more), importance (some events are more significant), and semantic similarity (what's actually related to the current question).

3. **Reflection** - This is the breakthrough. Before responding, the agent synthesizes retrieved memories into higher-level insights. "What do I actually know about this topic based on our conversation?" The reflection itself becomes a new memory, creating a hierarchy of understanding.

The Reflexion paper proved this works: agents with episodic memory reflection achieved 91% on HumanEval versus GPT-4's 80%. Not because they had more information, but because they *used* their information better.

For a practical chatbot, this means:
- Don't pass 20 turns of raw conversation to the LLM
- Retrieve the 3-5 most relevant turns
- Generate a 2-3 sentence reflection: "Based on our conversation, the user is asking about X in the context of Y"
- Use that reflection—not the raw history—for the response

---

## The Standalone Question: A Linguistic Anchor

The LangChain/LlamaIndex pattern of "standalone question reformulation" is elegant precisely because it's domain-agnostic:

> "Formulate a standalone question which can be understood without the chat history."

This single instruction handles every case:
- Pronouns: "What about them?" → "What about [entity from context]?"
- Demonstratives: "That sounds expensive" → "[Topic from context] sounds expensive"
- Ellipsis: "And the price?" → "What is the price of [topic from context]?"
- Meta-statements: "I meant the other one" → "Information about [corrected entity]"

The reformulation is purely syntactic. The LLM doesn't need to know that Texas Tech is a university or that quarterly earnings are financial data. It just needs to recognize that "them" is a pronoun requiring an antecedent.

This is why the CANARD dataset was so important. By collecting 40,527 examples of question rewrites, the researchers created a training signal that was entirely about *linguistic patterns*, not domain knowledge. The same model that rewrites "What about their defense?" for basketball can rewrite "What about their pricing?" for SaaS products.

---

## Confidence as Architecture

The Klarna case study is instructive. Their AI assistant handles two-thirds of customer service chats—not because it's smarter than humans, but because it knows when it's *not* smart enough.

The key was confidence calibration. When the AI was certain, it proceeded. When it wasn't, it escalated. This simple rule—"if uncertain, ask for help"—turned a potentially frustrating system into a reliable one.

The EMNLP 2023 research ("Just Ask for Calibration") revealed something surprising: for RLHF-trained models like GPT-4 and Claude, **verbalized confidence is more accurate than token probabilities**. When you ask the model to say "I'm 80% confident in this answer," that 80% is better calibrated than the softmax probability of the first token.

This suggests a four-tier approach:

| Confidence | Behavior |
|------------|----------|
| 90-100% | Proceed autonomously. The reformulation is certain. |
| 70-89% | Proceed but log for review. Probably correct, but monitor. |
| 40-69% | Request clarification. "Did you mean X or Y?" |
| 0-39% | Treat as new topic. Don't force context where none exists. |

The 70% and 90% thresholds aren't arbitrary—they come from industry benchmarks showing that well-designed chatbots achieve 70-80% containment rates for routine queries, with escalation for the rest.

---

## Routing: The Speed-Depth Tradeoff

Not every question deserves the same computational investment.

"Hi" doesn't need a web search. "What's 2+2?" doesn't need context resolution. "Explain the geopolitical implications of rare earth mineral supply chains" needs both—and probably multi-step reasoning.

Modern chatbot architecture requires a **router**—a lightweight classifier that triages queries before expensive operations:

```
Query → Router → [Simple | Context-Dependent | Complex]
                      ↓           ↓              ↓
                  Direct      Reformulate      Search +
                  Response    + Respond        Synthesize
```

For the simple path, response time should be under 100ms. For context resolution, under 200ms (using a fast model like Groq's Llama 3.1 8B). For complex queries with web search, 2-3 seconds is acceptable because the user expects depth.

Semantic caching adds another layer. If someone asks "What's the weather in Tokyo?" and someone else asks "Tokyo weather today?", the cosine similarity is above 0.8. Cache hit. No recomputation needed.

The goal isn't to make every response fast—it's to make *appropriate* responses fast and thorough responses thorough.

---

## Web Ability: Search as Grounding

A chatbot with web access is fundamentally different from one without. It can answer questions about current events, verify facts, and admit when it doesn't know something.

But web integration creates new challenges:

1. **When to search**: Not every question needs external information. "What's my name?" should come from conversation memory, not Google. The router must distinguish between memory queries, factual queries, and research queries.

2. **How to search**: The query the user speaks isn't always the query you should send to a search engine. "Tell me more about that" becomes useless without context resolution. The standalone question reformulation happens *before* the search.

3. **How to synthesize**: Search results are noisy. Ten blue links don't answer a question—they provide raw material for an answer. The chatbot must extract, filter, and synthesize. This is where reflection patterns help: "Based on these search results, what actually answers the user's question?"

4. **How to cite**: Users need to know where information came from. The response should link back to sources, not pretend the chatbot generated facts from thin air.

The RAG (Retrieval-Augmented Generation) pattern addresses this:
1. Reformulate query to standalone form
2. Retrieve relevant documents (web search or knowledge base)
3. Reflect on retrieved content ("What here is relevant?")
4. Generate response grounded in sources
5. Cite sources in the response

---

## The Anti-Patterns

Every best practice has a corresponding anti-pattern:

**Don't use domain-specific examples in prompts.** The moment you write "if the user mentions 'the score' or 'the coach', they're probably talking about sports," you've created a system that only works for sports.

**Don't keyword-match for context detection.** "It is raining" contains "it" but doesn't need context. "And the results?" has no pronouns but desperately needs context. Keyword matching is too crude.

**Don't dump full conversation history.** Context pollution is real. The model that receives 50 turns of history performs worse than the one that receives 3 relevant turns plus a reflection.

**Don't use a single confidence threshold.** The difference between 51% and 99% confidence is enormous. Treating them the same creates a brittle system.

**Don't ignore ellipsis.** Pronoun detection catches half of context-dependent questions. The other 20% are ellipsis—questions with missing subjects or objects that prior context supplies. "And the price?" is just as context-dependent as "What's their price?"

---

## The Synthesis

A modern AI chatbot with memory and web ability is not a single model—it's an orchestrated system:

1. **Sensory layer**: Captures the raw user input.

2. **Memory layer**: Maintains short-term context (last few turns), long-term facts (user preferences, topic cache), and episodic records (what happened in previous sessions).

3. **Context resolution layer**: Transforms the raw query into a standalone question using linguistic patterns, not domain knowledge. Outputs confidence score.

4. **Routing layer**: Based on query complexity and confidence, decides whether to respond directly, search the web, or request clarification.

5. **Retrieval layer**: When needed, searches web or knowledge base, retrieves relevant documents.

6. **Reflection layer**: Synthesizes retrieved information and conversation memory into a coherent understanding of what the user needs.

7. **Response layer**: Generates the answer, grounded in sources, calibrated in confidence.

8. **Memory write-back**: Updates the memory stream with this turn, potentially triggers reflection for long-term consolidation.

Each layer is informed by research:
- Context resolution: CANARD, CoQA, TREC CAsT
- Memory: Stanford Generative Agents, Reflexion
- Confidence: EMNLP 2023 calibration research
- Routing: Klarna case study, containment rate benchmarks

---

## The Core Principle

If there's one principle that unifies all of this, it's **linguistic universality over domain specificity**.

The reason LangChain's reformulation pattern works is that pronouns work the same way whether you're discussing quantum physics or pizza toppings. The reason confidence calibration works is that uncertainty is uncertainty, regardless of topic. The reason reflection improves performance is that synthesis is a general cognitive operation, not a domain-specific skill.

Build systems that understand *language*, and they'll understand every topic language can express.

Build systems that understand *sports*, and they'll only understand sports.

The research is clear. The patterns are proven. The path forward is to trust the linguistics and let go of the domains.

---

*This reflection synthesizes findings from ACL, EMNLP, TACL, UIST, and ICML publications spanning 2017-2024, alongside industry case studies from Klarna, LangChain, and LlamaIndex.*
