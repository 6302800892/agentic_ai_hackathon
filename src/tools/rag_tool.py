"""Agentic-RAG tool over the synthetic care-pathway / intake / coverage policy corpus (data/policy_corpus/).

Index  : Chroma (persistent, data/chroma/) + Sentence-Transformers `all-MiniLM-L6-v2` (local embeddings).
         Fallback: dependency-free keyword index (COPILOT_RAG_BACKEND=keyword, or if Chroma/model unavailable).
Tool   : `search_care_policy(query, pathway_hint=None, k=4) -> {"chunks": [...]}`
The retrieve -> grade -> rewrite -> retrieve loop lives in src/agents/care_pathway.py (retrieval-in-the-loop).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import DATA, get_settings

log = logging.getLogger(__name__)

CORPUS_DIR = DATA / "policy_corpus"
CHROMA_DIR = DATA / "chroma"
COLLECTION = "care_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOOL_NAME = "search_care_policy"


class RagInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    pathway_hint: Optional[str] = Field(None, description="MSK, DERM, CARDIO, MENTAL_HEALTH, PEDS, GENERAL, ADMIN")
    k: int = Field(4, ge=1, le=10)


class RagChunk(BaseModel):
    chunk_id: str
    policy_id: str
    section: str
    title: str
    pathway: str
    text: str
    score: float


class RagOutput(BaseModel):
    chunks: list[RagChunk]
    backend: str


# --------------------------------------------------------------------------- corpus loading / chunking
_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def load_chunks() -> list[dict]:
    """Chunk each policy by `## §n` section, keeping policy metadata for citations."""
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = _FM_RE.match(raw)
        meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line) if m else {}
        body = raw[m.end():] if m else raw
        pid = meta.get("policy_id", path.stem)
        for sec in re.split(r"\n(?=## )", body):
            head = re.match(r"## (§\d+)\s*(.*)", sec)
            if not head:
                continue
            section = head.group(1)
            chunks.append({"chunk_id": f"{pid}#{section}", "policy_id": pid, "section": section,
                           "title": f"{meta.get('title', pid)} - {head.group(2).strip()}",
                           "pathway": meta.get("pathway", "GENERAL"), "text": sec.strip()})
    return chunks


GENERAL_PATHWAYS = {"INTAKE", "COVERAGE", "REFERRAL", "ESCALATION"}


class KeywordIndex:
    backend = "keyword"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.docs = [Counter(self._tok(c["title"] + " " + c["text"])) for c in chunks]
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + len(chunks) / n) for t, n in df.items()}

    @staticmethod
    def _tok(text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]

    def search(self, query: str, k: int, pathway_hint: str | None) -> list[dict]:
        q = self._tok(query)
        scored = []
        for c, d in zip(self.chunks, self.docs):
            s = sum(self.idf.get(t, 0) * (1 + math.log(d[t])) for t in q if d.get(t))
            if pathway_hint and c["pathway"] == pathway_hint.upper():
                s *= 1.5
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        top = max(scored[0][0], 1e-9) if scored else 1
        return [{**c, "score": round(s / top, 4)} for s, c in scored[:k] if s > 0]


def embedding_function():
    """Sentence-Transformers all-MiniLM-L6-v2; if torch/sklearn cannot load, the same model via Chroma's ONNX runtime."""
    from chromadb.utils import embedding_functions as ef
    try:
        import sentence_transformers  # noqa: F401  (fail fast if blocked)
        return ef.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL), f"sentence-transformers/{EMBED_MODEL}"
    except Exception as e:
        log.warning("sentence-transformers unavailable (%s); using ONNX %s via chromadb.", e, EMBED_MODEL)
        return ef.DefaultEmbeddingFunction(), f"onnx/{EMBED_MODEL}"


class ChromaIndex:
    backend = "chroma"

    def __init__(self):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef, self.embedder = embedding_function()
        self.col = self.client.get_or_create_collection(COLLECTION, embedding_function=self.ef,
                                                        metadata={"hnsw:space": "cosine"})
        if self.col.count() == 0:
            self.build()

    def build(self) -> int:
        chunks = load_chunks()
        existing = self.col.get()["ids"]
        if existing:
            self.col.delete(ids=existing)
        self.col.add(ids=[c["chunk_id"] for c in chunks], documents=[c["text"] for c in chunks],
                     metadatas=[{k: c[k] for k in ("policy_id", "section", "title", "pathway")} for c in chunks])
        return len(chunks)

    def _query(self, query: str, n: int, where: dict | None = None) -> list[dict]:
        res = self.col.query(query_texts=[query], n_results=n, where=where)
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"chunk_id": cid, "text": doc, **meta, "score": round(1 - float(dist), 4)})
        return out

    def search(self, query: str, k: int, pathway_hint: str | None) -> list[dict]:
        hits = {h["chunk_id"]: h for h in self._query(query, min(k * 2, 12))}
        if pathway_hint:
            try:
                for h in self._query(query, 2, where={"pathway": pathway_hint.upper()}):
                    h["score"] = round(h["score"] + 0.1, 4)  # boost the hinted pathway
                    hits[h["chunk_id"]] = h
            except Exception:
                pass
        return sorted(hits.values(), key=lambda h: -h["score"])[:k]


@lru_cache(maxsize=2)
def get_index(backend: str | None = None):
    backend = backend or get_settings().rag_backend
    if backend == "chroma":
        try:
            return ChromaIndex()
        except Exception as e:  # graceful degradation
            log.warning("Chroma/Sentence-Transformers unavailable (%s); using keyword index.", e)
    return KeywordIndex(load_chunks())


def build_rag_tool(index=None) -> StructuredTool:
    index = index or get_index()

    async def search_care_policy(query: str, pathway_hint: Optional[str] = None, k: int = 4) -> dict:
        hits = index.search(query, k, pathway_hint)
        return RagOutput(chunks=[RagChunk(**{f: h[f] for f in RagChunk.model_fields}) for h in hits],
                         backend=index.backend).model_dump()

    return StructuredTool.from_function(
        coroutine=search_care_policy, name=TOOL_NAME, args_schema=RagInput,
        description="Search the care-pathway, intake, referral and coverage policy corpus. Returns cited chunks.")
