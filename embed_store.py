"""
Embedding + vector store for The Unofficial Guide (Milestone 4).

  - Embeds every chunk from pipeline.build_chunks() with all-MiniLM-L6-v2.
  - Stores them in a local, persistent ChromaDB collection with metadata
    (source filename + chunk index) for later citation.
  - Exposes retrieve(query, k) for semantic search.

Build/refresh the index:
    python embed_store.py            # (re)builds the index, then runs test queries
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

from pipeline import build_chunks

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "unofficial_guide"

# Load the embedding model once at import (runs locally, no API key).
_model = SentenceTransformer(EMBED_MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_DIR)


def _embed(texts):
    """Return embeddings as plain Python lists (what Chroma expects)."""
    return _model.encode(texts, show_progress_bar=False).tolist()


def build_index():
    """(Re)build the Chroma collection from scratch so reruns don't duplicate."""
    # Drop any existing collection so we always start clean.
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance for sentence embeddings
    )

    chunks = build_chunks()
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]}
                 for c in chunks]
    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=_embed(documents),
        metadatas=metadatas,
    )
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' "
          f"at {CHROMA_DIR}")
    return collection


def get_collection():
    """Return the existing collection (build it first if missing/empty)."""
    try:
        collection = _client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    return build_index()


def retrieve(query, k=4):
    """Semantic search. Returns a list of dicts: {text, source, chunk_index, distance}."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=_embed([query]),
        n_results=k,
    )
    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return hits


if __name__ == "__main__":
    build_index()

    # Test retrieval on 3 of the 5 evaluation questions (Milestone 4 checkpoint).
    test_queries = [
        "How much do students pay for rent near campus?",
        "How do I make sure I get my security deposit back?",
        "How can I avoid rental scams when apartment hunting?",
    ]
    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        for i, hit in enumerate(retrieve(q, k=4), 1):
            print(f"\n  #{i}  distance={hit['distance']:.3f}  "
                  f"source={hit['source']} (chunk {hit['chunk_index']})")
            preview = hit["text"][:220].replace("\n", " ")
            print(f"      {preview}...")
