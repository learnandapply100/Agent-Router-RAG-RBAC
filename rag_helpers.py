"""
rag_helpers.py — Helper functions for the Agentic RAG with Semantic Cache notebook
(003. Agentic Router_semantic_caching_rbac.ipynb).

All heavy lifting lives here so the notebook stays clean and focused
on demonstrating the system behaviour.

Quick start in a Colab/Jupyter notebook:
    import sys, nest_asyncio
    sys.path.insert(0, '/content/multi-agent-course/modules/Module_3_Production_Agentic_RAG_AI_Systems')
    nest_asyncio.apply()

    from rag_helpers import init_rag, SemanticCaching, agentic_rag_with_cache

    init_rag(openai_api_key="...", serp_api_key="...", qdrant_path="...")
    cache = SemanticCaching(clear_on_init=True)
    agentic_rag_with_cache("What was Uber's revenue in 2021?", cache)

What's in this file:
    1. init_rag()                 — one-time setup of the shared clients and models
    2. SemanticCaching            — the FAISS cache (lookup + storage only)
    3. Pipeline pieces            — router, web search, Qdrant retrieval, answer generation
    4. agentic_rag_with_cache()   — the main entry point that ties 2 and 3 together
"""

# NOTE: import sentence_transformers BEFORE faiss.
# Both ship their own OpenMP runtime; on macOS, loading a SentenceTransformer
# after faiss aborts the process with no traceback (the kernel just dies).
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

import faiss
import json
import re
import time
import asyncio

import numpy as np
import requests
import torch

from openai import OpenAI, OpenAIError
import qdrant_client as _qdrant_lib


# ── Settings ──────────────────────────────────────────────────────────────────

# One model name for every LLM call (routing + answer generation).
# The notebook's RBAC section reuses this constant too, so change it in one place.
LLM_MODEL = "gpt-5.6-luna"

# Qdrant collection used for each route label.
COLLECTIONS = {
    "OPENAI_QUERY":       "opnai_data",   # OpenAI Agents documentation
    "10K_DOCUMENT_QUERY": "10k_data",     # Uber 2021 + Lyft 2024 10-K filings (both in one collection)
}

ROUTES = ("OPENAI_QUERY", "10K_DOCUMENT_QUERY", "INTERNET_QUERY")


# ── Module-level shared state (populated by init_rag) ────────────────────────
_openaiclient   = None
_serp_api_key   = None
_qdrant         = None
_text_tokenizer = None
_text_model     = None


class RAGError(Exception):
    """
    Raised when a backend fails to produce a real answer (network error,
    empty search results, retrieval failure, ...).

    Failures are raised instead of returned as text so that the caller can
    tell an answer apart from an error — and never store an error in the cache.
    """


# ── 1. Initialisation ─────────────────────────────────────────────────────────

def init_rag(openai_api_key: str, serp_api_key: str, qdrant_path: str) -> None:
    """
    Initialise all shared state for the Agentic RAG pipeline.

    Must be called once before using any other function in this module.

    Args:
        openai_api_key: OpenAI API key — used for routing and RAG generation.
        serp_api_key:   SerpApi key  — used for live Google search results.
        qdrant_path:    Absolute path to the local Qdrant vector database
                        (contains 'opnai_data' and '10k_data' collections).
    """
    global _openaiclient, _serp_api_key, _qdrant, _text_tokenizer, _text_model

    _openaiclient = OpenAI(api_key=openai_api_key)
    _serp_api_key = serp_api_key
    _qdrant = _qdrant_lib.AsyncQdrantClient(path=qdrant_path)

    # Why a second copy of the Nomic model (the cache loads its own via SentenceTransformer)?
    # The Qdrant collections were built with *these* embeddings: raw AutoModel output,
    # mean-pooled, not normalised. A query must be embedded the same way as the stored
    # chunks or the search results get worse. So the cache and the retriever each use
    # the embedding style that matches their own index. Don't "unify" them unless you
    # also rebuild the Qdrant collections.
    print("Loading Nomic text model for Qdrant retrieval embeddings...")
    _text_tokenizer = AutoTokenizer.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    _text_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    _text_model.eval()  # inference mode (no dropout)
    print("✅ RAG pipeline ready.")


# ── 2. SemanticCaching ────────────────────────────────────────────────────────

class SemanticCaching:
    """
    FAISS-backed semantic cache with a time-sensitivity filter.

    The class only does lookup and storage. The decision of what to do on a
    hit or miss lives in agentic_rag_with_cache():
        is_time_sensitive → True   →  skip cache, run the pipeline, don't store
        check_cache       → HIT    →  return stored answer instantly ⚡
        check_cache       → MISS   →  run the pipeline, store answer, return
    """

    # Matched as WHOLE words/phrases (see is_time_sensitive), so "now" does not
    # match "know" and "live" does not match "deliver".
    TIME_SENSITIVE_KEYWORDS = [
        "today", "tonight", "now", "currently", "current",
        "latest", "recent", "recently", "right now", "at the moment",
        "at present", "as of now", "this week", "this month", "this year",
        "this quarter", "this season", "this morning", "this afternoon",
        "this evening", "this weekend", "yesterday", "tomorrow",
        "last week", "last month", "last year", "upcoming", "live",
        "breaking", "just happened", "what time", "what day", "what date",
        "happening now", "events today", "news today", "news this week",
        "stock price", "stock prices", "share price", "share prices",
        "weather", "forecast", "temperature",
        "real-time", "realtime", "schedule today", "outage", "outages",
    ]

    def __init__(
        self,
        json_file: str = "rag_cache.json",
        threshold: float = 0.30,
        clear_on_init: bool = False,
    ):
        """
        Args:
            json_file:      Path to the JSON file used for cache persistence.
            threshold:      Max squared L2 distance for a cache hit — lower is stricter.
                            0.30 separates real paraphrases (~0.16–0.25 on the course's
                            demo questions) from merely related questions (~0.38+).
                            Measure this on your own queries; it is a product decision.
            clear_on_init:  If True, wipe any existing cache on startup.
        """
        self.embedding_dim = 768
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.euclidean_threshold = threshold
        self.json_file = json_file

        # One regex for all keywords, with word boundaries on both sides.
        self._time_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(k) for k in self.TIME_SENSITIVE_KEYWORDS) + r")\b"
        )

        print("Loading Nomic embedding model for semantic cache...")
        self.encoder = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        print("Cache embedding model ready.")

        if clear_on_init:
            self.clear_cache()
        else:
            self.load_cache()

    # ── Time-sensitivity ─────────────────────────────────────────────────────

    def is_time_sensitive(self, question: str) -> bool:
        """
        Return True if the question contains a time word whose answer changes
        over time. Such answers are never read from or written to the cache.

        Examples that return True  → bypass cache, always answered fresh:
            'What is the current stock price of AAPL?'
            'What are the latest AI news this week?'
            'Are there any AWS outages right now?'

        Examples that return False → safe to cache:
            'What was Uber revenue in 2021?'
            'How do OpenAI Agents work?'
            'How do I know which model to use?'   ← "know" is not "now"
        """
        return bool(self._time_pattern.search(question.lower()))

    # ── Persistence ───────────────────────────────────────────────────────────

    def clear_cache(self):
        """Reset in-memory state and overwrite the JSON file."""
        self.cache = {"questions": [], "embeddings": [], "response_text": []}
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.save_cache()
        print("Semantic cache cleared.")

    def load_cache(self):
        """Load entries from JSON and rebuild the FAISS index from the saved embeddings."""
        try:
            with open(self.json_file, "r") as f:
                self.cache = json.load(f)
            # Row i in FAISS must line up with entry i in the JSON lists,
            # so re-add every saved embedding in its original order.
            if self.cache["embeddings"]:
                vecs = np.array(self.cache["embeddings"], dtype=np.float32)
                self.index.add(vecs)
            print(f"Cache loaded: {len(self.cache['questions'])} entries.")
        except FileNotFoundError:
            self.cache = {"questions": [], "embeddings": [], "response_text": []}
            print("No existing cache found — starting fresh.")

    def save_cache(self):
        """Persist the current cache to disk."""
        with open(self.json_file, "w") as f:
            json.dump(self.cache, f)

    # ── Lookup & storage ──────────────────────────────────────────────────────

    def check_cache(self, question: str):
        """
        Encode the question and search the FAISS index for the nearest stored question.

        Returns:
            tuple: (hit, answer, embedding, similarity, row_id)
                hit        (bool)          — True if a cached answer was found
                answer     (str | None)    — The cached answer, or None on miss
                embedding  (np.ndarray)    — Computed embedding; reused on a miss so the
                                             question isn't encoded twice
                similarity (float | None)  — Cosine similarity to the matched question
                                             (1.0 = identical), or None on miss
                row_id     (int | None)    — Cache row index, or None on miss
        """
        embedding = self.encoder.encode([question], normalize_embeddings=True)

        if self.index.ntotal == 0:
            return False, None, embedding, None, None

        D, I = self.index.search(embedding, 1)
        distance = float(D[0][0])  # squared L2 distance
        if I[0][0] != -1 and distance <= self.euclidean_threshold:
            row_id = int(I[0][0])
            # For normalised vectors: squared L2 = 2 - 2·cosine  →  cosine = 1 - d/2
            similarity = 1.0 - distance / 2.0
            return True, self.cache["response_text"][row_id], embedding, similarity, row_id

        return False, None, embedding, None, None

    def add_to_cache(self, question: str, answer: str, embedding: np.ndarray):
        """Store a new question-answer pair and persist to disk."""
        self.cache["questions"].append(question)
        self.cache["embeddings"].append(embedding[0].tolist())
        self.cache["response_text"].append(answer)
        self.index.add(embedding)
        self.save_cache()


# ── 3. Agentic RAG pipeline pieces ───────────────────────────────────────────

def get_internet_content(user_query: str, action: str = "INTERNET_QUERY") -> str:
    """
    Live Google search via SerpApi.

    Used whenever the router picks INTERNET_QUERY (whether or not the question
    is time-sensitive). Results are an answer-box snippet + top organic results.

    Args:
        user_query: The user's question.
        action:     Route label (unused here, kept for signature compatibility).

    Returns:
        str: Formatted search results.

    Raises:
        RAGError: if the key is missing, the request fails, or nothing is found.
    """
    print("Getting your response from the internet 🌐 ...")
    if not _serp_api_key:
        raise RAGError("SERP_API_KEY not set — call init_rag() first.")

    params = {"q": user_query, "api_key": _serp_api_key, "engine": "google", "num": 5}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        raise RAGError(f"SerpApi request error: {e}") from e

    parts = []
    ab = data.get("answer_box", {})
    if ab.get("answer"):
        parts.append(f"[Direct Answer] {ab['answer']}")
    elif ab.get("snippet"):
        parts.append(f"[Direct Answer] {ab['snippet']}")

    for i, r in enumerate(data.get("organic_results", [])[:5], 1):
        if r.get("snippet"):
            parts.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    {r['snippet']}\n"
                f"    Source: {r.get('link', '')}"
            )

    if not parts:
        raise RAGError("No search results found.")
    return "\n\n".join(parts)


def route_query(user_query: str) -> dict:
    """
    Ask the LLM (LLM_MODEL) which knowledge source should answer the question.

    Returns:
        dict with keys 'action' and 'reason'.
        action is one of: 'OPENAI_QUERY', '10K_DOCUMENT_QUERY', 'INTERNET_QUERY'.
        If anything goes wrong, it falls back to INTERNET_QUERY.
    """
    if not _openaiclient:
        return {"action": "INTERNET_QUERY", "reason": "RAG not initialised — call init_rag()."}

    prompt = f"""
    Classify the user query into exactly one of three categories:
    1. "OPENAI_QUERY"       — Questions about OpenAI documentation: agents, APIs, models, embeddings.
    2. "10K_DOCUMENT_QUERY" — Questions about company financials or 10-K filings (Uber, Lyft).
    3. "INTERNET_QUERY"     — Everything else: general knowledge, trends, comparisons, real-time data.

    Respond ONLY with this JSON (no other text):
    {{
        "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "INTERNET_QUERY",
        "reason": "brief justification"
    }}

    User: {user_query}
    """
    try:
        response = _openaiclient.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
        )
        content = response.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        route = json.loads(match.group())
    except (OpenAIError, json.JSONDecodeError, AttributeError) as e:
        return {"action": "INTERNET_QUERY", "reason": f"Routing error: {e}"}

    if route.get("action") not in ROUTES:
        return {"action": "INTERNET_QUERY", "reason": f"Unknown route {route.get('action')!r} — defaulting to internet."}
    return route


def _get_text_embeddings(text: str) -> np.ndarray:
    """
    Mean-pooled token embeddings used for Qdrant similarity search.
    (Same method used to build the Qdrant collections — see init_rag().)
    """
    inputs = _text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # inference only — no gradients needed, saves memory
        outputs = _text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()


def _format_context(points) -> str:
    """
    Number each retrieved chunk and label it with whatever source info the payload has,
    so the LLM can cite it as [1], [2], ... and the reader can tell where it came from.
    """
    blocks = []
    for i, p in enumerate(points, 1):
        meta = p.payload.get("metadata") or {}
        label_parts = [str(meta[k]) for k in ("company", "fiscal_year", "source", "title") if meta.get(k)]
        label = f" ({', '.join(label_parts)})" if label_parts else ""
        blocks.append(f"[{i}]{label}\n{p.payload['content']}")
    return "\n\n".join(blocks)


def _rag_formatted_response(user_query: str, context: str) -> str:
    """Generate an answer grounded ONLY in the numbered context chunks, with [n] citations."""
    prompt = f"""
    Answer the user's question using ONLY the numbered context below.
    Cite the chunks you used with their numbers, e.g. [1] or [1][3].
    If the context does not contain the answer, say so plainly instead of guessing.

    Context:
    {context}

    Question: {user_query}
    """
    response = _openaiclient.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": prompt}],
    )
    return response.choices[0].message.content


async def _retrieve_and_respond(user_query: str, action: str) -> str:
    """
    Embed query → search the matching Qdrant collection (top 3 chunks) → generate a grounded answer.

    Raises:
        RAGError: if the action is invalid, retrieval fails, or nothing relevant is found.
    """
    if action not in COLLECTIONS:
        raise RAGError(f"Invalid action: {action}")
    try:
        embedding = _get_text_embeddings(user_query)
        hits = await _qdrant.query_points(
            collection_name=COLLECTIONS[action], query=embedding, limit=3
        )
    except Exception as e:
        raise RAGError(f"Retrieval error: {e}") from e

    if not hits.points:
        raise RAGError("No relevant content found in the database.")

    try:
        return _rag_formatted_response(user_query, _format_context(hits.points))
    except OpenAIError as e:
        raise RAGError(f"Answer generation error: {e}") from e


def _run_rag_pipeline(user_query: str):
    """
    Route the query and call the matching handler.

    Returns:
        tuple: (text, ok)
            text (str)  — the answer, or an error message if ok is False
            ok   (bool) — True only for a real answer. agentic_rag_with_cache()
                          uses this to make sure errors are never cached.
    """
    GREY, RESET = "\033[90m", "\033[0m"

    route = route_query(user_query)
    action = route.get("action", "INTERNET_QUERY")
    reason = route.get("reason", "")
    print(f"{GREY}📍 Route: {action}  |  {reason}{RESET}")

    handlers = {
        "OPENAI_QUERY":       lambda q: asyncio.run(_retrieve_and_respond(q, "OPENAI_QUERY")),
        "10K_DOCUMENT_QUERY": lambda q: asyncio.run(_retrieve_and_respond(q, "10K_DOCUMENT_QUERY")),
        "INTERNET_QUERY":     lambda q: get_internet_content(q),
    }

    try:
        return handlers[action](user_query), True
    except RAGError as e:
        return f"⚠️ {e}", False
    except Exception as e:
        return f"⚠️ Execution error: {e}", False


# ── 4. Public entry point ─────────────────────────────────────────────────────

def agentic_rag_with_cache(user_query: str, cache: SemanticCaching) -> str:
    """
    Agentic RAG with a semantic cache layer.

    Query flow:
        1. Time-sensitive?  → skip the cache, run the full pipeline (router → source), don't store
        2. Cache HIT        → return stored answer instantly ⚡
        3. Cache MISS       → run the full pipeline, store the answer (only if it succeeded), return

    Args:
        user_query: The user's question.
        cache:      A SemanticCaching instance to check and update.

    Returns:
        str: The final answer text (or an error message, which is never cached).
    """
    CYAN, GREEN, YELLOW, RED, BOLD, RESET = (
        "\033[96m", "\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"
    )

    print(f"{BOLD}{CYAN}👤 Query:{RESET} {user_query}\n")
    start = time.time()

    # 1. Time-sensitivity bypass — the router still picks the source; we just don't cache
    if cache.is_time_sensitive(user_query):
        print(f"{YELLOW}⏰ Time-sensitive — bypassing cache for a fresh answer.{RESET}\n")
        result, _ = _run_rag_pipeline(user_query)
        print(f"\n{BOLD}{CYAN}🤖 Response (live, {time.time() - start:.2f}s):{RESET}\n{result}\n")
        return result

    # 2. Semantic cache lookup
    hit, cached_answer, embedding, similarity, row_id = cache.check_cache(user_query)

    if hit:
        print(
            f"{GREEN}✅ Cache HIT{RESET} "
            f"(row {row_id}, cosine similarity: {similarity:.3f}, {time.time() - start:.3f}s)\n"
        )
        print(f"{BOLD}{CYAN}🤖 Response (cached):{RESET}\n{cached_answer}\n")
        return cached_answer

    # 3. Cache miss → full pipeline
    print(f"{YELLOW}❌ Cache MISS — running Agentic RAG pipeline...{RESET}\n")
    result, ok = _run_rag_pipeline(user_query)

    if ok:
        cache.add_to_cache(user_query, result, embedding)
        print(f"\n{GREEN}💾 Cached for future similar queries.{RESET}")
    else:
        print(f"\n{RED}⚠️ Pipeline failed — NOT cached, so the next attempt tries again.{RESET}")

    print(f"\n{BOLD}{CYAN}🤖 Response ({time.time() - start:.2f}s):{RESET}\n{result}\n")
    return result
