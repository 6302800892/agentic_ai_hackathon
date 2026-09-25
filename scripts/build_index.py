"""(Re)build the Chroma vector index over data/policy_corpus/ (regenerable; data/chroma/ is git-ignored).

    python scripts/build_index.py            # chroma + sentence-transformers (ONNX MiniLM fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.rag_tool import ChromaIndex, load_chunks  # noqa: E402


def main() -> None:
    chunks = load_chunks()
    idx = ChromaIndex()
    n = idx.build()
    print(f"indexed {n} chunks from {len({c['policy_id'] for c in chunks})} policies with {idx.embedder}")
    hits = idx.search("physiotherapy for back pain", 3, "MSK")
    print("sanity query:", [h["chunk_id"] for h in hits])


if __name__ == "__main__":
    main()
