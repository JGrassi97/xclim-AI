# xclim_tools.rag.XclimRAGAgent
# ====================================================
# Graph-based retrieval-augmented generation (RAG) agent for **xclim** indicators
# using LangGraph. Replaces the previous `XclimRAGAgent` with an explicit control
# flow in a StateGraph with 4 nodes:
#
# 1. query_refiner  → LLM rewrites the user query and computes its embedding
# 2. rag_executor   → retrieves top-k indicators and updates covered topics
# 3. evaluator      → determines whether another round is needed
# 4. aggregator     → selects and scores final indicators
#
# The loop continues until:
# - iter_idx == max_iters
# - mean_score ≥ score_threshold and all topics are covered

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Set, TypedDict
import json
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph
from langchain_chroma import Chroma

from xclim_ai.utils.llm import initialize_embeddings, initialize_llm_rag
from xclim_ai.utils.prompts import load_prompt
from xclim_ai.utils.paths import XCLIM_CHROMA
from xclim_ai.utils.tools import get_valid_tools
from xclim_ai.rag.utils import safe_get_embeddings, get_top_k
from xclim_ai.utils.logging import get_logger, set_logger_level


@dataclass
class RetrievalRound:
    """One retrieval step in the iterative process."""
    refined_query: str
    indicators: List[Dict[str, Any]]
    score: float  # mean cosine similarity for the round


class RagState(TypedDict, total=False):
    original_query: str
    history: List[RetrievalRound]
    current_query: str
    refined_query: str
    query_embedding: np.ndarray
    remaining_topics: List[str]
    iter_idx: int
    mean_score: float
    last_hits: List[Dict[str, Any]]
    continue_: bool
    final_hits: List[Dict[str, Any]]
    best_query: str


@dataclass
class XclimRAGAgent:
    k: int = 2
    max_iters: int = 4
    score_threshold: float = 0.75
    return_k: int | None = None
    verbose: bool = False

    _llm: Any = field(init=False, repr=False)
    _embeddings: Any = field(init=False, repr=False)
    _store: Chroma = field(init=False, repr=False)
    _ids: Sequence[str] = field(init=False, repr=False)
    _emb_matrix: np.ndarray = field(init=False, repr=False)

    _query_prompt: str = field(init=False, repr=False)
    _eval_prompt: str = field(init=False, repr=False)
    _topic_prompt: str = field(init=False, repr=False)

    _graph: Any = field(default=None, init=False, repr=False)
    _topic_cache: Set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = get_logger()
        set_logger_level(self._logger, self.verbose)

        self._llm = initialize_llm_rag()
        self._embeddings = initialize_embeddings()

        valid_tools = [cls(ds=None) for cls in get_valid_tools()]
        valid_tools_names = [tool.name for tool in valid_tools]

        self._store = Chroma(persist_directory=str(XCLIM_CHROMA), embedding_function=self._embeddings)
        self._ids, self._emb_matrix = safe_get_embeddings(self._store, valid_tools_names=valid_tools_names)

        self._query_prompt = load_prompt("rag_refiner.md")
        self._eval_prompt = load_prompt("result_evaluator.md")
        self._topic_prompt = load_prompt("topic_extractor.md")

        if self.return_k is None:
            self.return_k = self.k

        self._graph = self._build_graph()

    def run(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        self._topic_cache = set(self._extract_topics_llm(user_query))

        init_state: RagState = {
            "original_query": user_query,
            "current_query": user_query,
            "remaining_topics": list(self._topic_cache),
            "iter_idx": 0,
            "mean_score": 0.0,
            "history": [],
        }

        final_state: RagState = self._graph.invoke(init_state, config={"recursion_limit": 100})
        best_q = final_state.get("best_query", final_state["current_query"])
        hits = final_state["final_hits"]
        return best_q, hits

    def _build_graph(self):
        sg = StateGraph(RagState)

        sg.add_node("query_refiner", self._query_refiner)
        sg.add_node("rag_executor", self._rag_executor)
        sg.add_node("evaluator", self._evaluator)
        sg.add_node("aggregator", self._aggregator)

        sg.set_entry_point("query_refiner")
        sg.add_edge("query_refiner", "rag_executor")
        sg.add_edge("rag_executor", "evaluator")

        def _should_continue(state: RagState) -> bool:
            return bool(state.get("continue_", False))

        sg.add_conditional_edges("evaluator", _should_continue, {True: "query_refiner", False: "aggregator"})
        return sg.compile()

    def _query_refiner(self, state: RagState) -> RagState:
        q = state["current_query"]
        topics = state.get("remaining_topics", [])
        prompt = self._query_prompt.replace("{{MISSING_TOPICS}}", topics[0] if topics else "none")
        refined = self._llm.invoke(f"{q}\n\n{prompt}").content.strip()
        emb = np.asarray(self._embeddings.embed_query(refined), dtype=float)

        if self.verbose:
            self._logger.debug("\n[Query Refinement]")
            self._logger.debug(f"Original: {q}")
            self._logger.debug(f"Refined: {refined}")
            self._logger.debug(f"Remaining topics: {topics}")

        state.update({"refined_query": refined, "query_embedding": emb})
        return state

    def _rag_executor(self, state: RagState) -> RagState:
        emb = state["query_embedding"]
        sim = cosine_similarity(emb.reshape(1, -1), self._emb_matrix)
        hits = get_top_k(sim, self._ids, self._store, k=self.k)
        mean = float(np.mean([h["similarity"] for h in hits])) if hits else 0.0

        state["history"].append(RetrievalRound(state["refined_query"], hits, mean))
        remaining = set(state["remaining_topics"])
        remaining -= self._topics_covered(hits)

        if self.verbose:
            self._logger.debug("\n[RAG Execution]")
            self._logger.debug(f"Top-{self.k} mean similarity: {mean:.3f}")
            self._logger.debug(f"Retrieved: {[h['id'] for h in hits]}")
            self._logger.debug(f"Remaining topics: {remaining}")

        state.update({
            "remaining_topics": list(remaining),
            "mean_score": mean,
            "iter_idx": state["iter_idx"] + 1,
            "last_hits": hits,
        })
        return state

    def _evaluator(self, state: RagState) -> RagState:
        cont = ((state["mean_score"] < self.score_threshold or state["remaining_topics"]) and
                state["iter_idx"] < self.max_iters)
        state["continue_"] = cont

        if self.verbose:
            self._logger.debug("\n[Evaluator]")
            self._logger.debug(f"Score: {state['mean_score']:.3f}")
            self._logger.debug(f"Remaining: {state['remaining_topics']}")
            self._logger.debug(f"Continue? {cont}")

        if cont:
            missing = ", ".join(sorted(state["remaining_topics"])) or "(none)"
            prompt = (
                f"The following climate-indicator topics were not satisfied: {missing}.\n"
                "Rewrite the query to focus only on these missing topics.\n\n"
                f"Current query:\n{state['refined_query']}"
            )
            state["current_query"] = self._llm.invoke(prompt).content.strip()

        return state

    def _aggregator(self, state: RagState) -> RagState:
        unique: Dict[str, Dict[str, Any]] = {}
        for rnd in state["history"]:
            for h in rnd.indicators:
                unique[h["id"]] = h

        scored = self._score_hits_with_llm(state["original_query"], list(unique.values()))
        final_hits = self._diversified_select(scored)[: self.return_k]

        if self.verbose:
            self._logger.debug("\n[Aggregator]")
            self._logger.debug("Selected indicators:")
            for h in final_hits:
                self._logger.debug("- %s (%.3f)", h["id"], h.get("_llm_score", 0.0))

        state.update({
            "final_hits": final_hits,
            "best_query": state.get("refined_query", state["current_query"]),
        })
        return state

    def _extract_topics_llm(self, q: str) -> List[str]:
        try:
            raw = self._llm.invoke(f"User query:\n\"\"\"\n{q}\n\"\"\"\n\n{self._topic_prompt}").content.strip()
            if self.verbose:
                self._logger.debug("\n[Topic Extraction]")
                self._logger.debug(f"Extracted raw: {raw}")
            return [t.lower() for t in json.loads(raw) if isinstance(t, str) and t.strip()]
        except Exception:
            return self._extract_topics_regex(q)

    @staticmethod
    def _extract_topics_regex(q: str) -> List[str]:
        parts = re.split(r"\\b(?:and|or|,|;|/)\\b", q, flags=re.I)
        return [" ".join(p.split()).lower() for p in parts if p.strip()]

    def _topics_covered(self, h: Dict[str, Any]) -> Set[str]:
        if isinstance(h, list):
            excerpt = str([x["id"] for x in h])
        else:
            excerpt = h["id"]

        covered: Set[str] = set()
        for topic in self._topic_cache:
            prompt = (
                f'Topic: "{topic}"\n\n'
                f'Indicator excerpt:\n"""\n{excerpt}\n"""\n\n'
                "Does the indicator explicitly address this topic? Answer with 'yes' or 'no'."
            )
            try:
                reply = self._llm.invoke(prompt).content.strip().lower()
                if reply.startswith("y"):
                    covered.add(topic)
            except Exception:
                if re.search(rf"\\b{re.escape(topic)}\\b", excerpt.lower()):
                    covered.add(topic)
        return covered

    def _score_hits_with_llm(self, q: str, hits: List[Dict[str, Any]]):
        for h in hits:
            excerpt = h.get("text", "")[:800]
            prompt = (f"User query:\n\"\"\"\n{q}\n\"\"\"\n\n"
                      f"Candidate indicator:\n\"\"\"\n{excerpt}\n\"\"\"\n\n{self._eval_prompt}")
            try:
                h["_llm_score"] = h.get("similarity", 0.0)
            except Exception:
                h["_llm_score"] = h.get("similarity", 0.0)

            if self.verbose:
                self._logger.debug("Scoring: %s → %.3f", h["id"], h["_llm_score"])

        return hits

    def _diversified_select(self, hits: List[Dict[str, Any]]):
        sel: List[Dict[str, Any]] = []
        for topic in sorted(self._topic_cache):
            cand = [h for h in hits if topic in self._topics_covered(h) and h not in sel]
            if cand:
                sel.append(max(cand, key=lambda x: x["_llm_score"]))
                if len(sel) >= self.return_k:
                    return sel

        remaining = [h for h in hits if h not in sel]
        remaining.sort(key=lambda x: -x["_llm_score"])
        sel += remaining[: self.return_k - len(sel)]
        return sel