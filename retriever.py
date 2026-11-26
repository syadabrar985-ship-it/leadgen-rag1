import re
from typing import Any
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 keyword search."""
    return re.findall(r"[a-z0-9]+(?:[./_-][a-z0-9]+)*", text.lower())


class HybridSOPRetriever:
    """Combines ChromaDB semantic search, BM25 keyword search, and RRF hybrid fusion."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_model_name: str,
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: list[str] = []
        self._corpus_docs: list[str] = []
        self._corpus_meta: list[dict[str, Any]] = []
        self._rebuild_bm25()

    def doc_count(self) -> int:
        """Return total indexed chunks."""
        return len(self._corpus_docs)

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert chunks into ChromaDB and rebuild BM25."""
        if not chunks:
            raise ValueError("No chunks provided to index.")

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Load stored chunks from ChromaDB to rebuild the BM25 index."""
        stored = self.collection.get(include=["documents", "metadatas"])
        self._corpus_ids = list(stored.get("ids") or [])
        self._corpus_docs = list(stored.get("documents") or [])
        self._corpus_meta = [dict(m) if m else {} for m in (stored.get("metadatas") or [])]

        tokenized = [_tokenize(doc) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    def semantic_search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Retrieve top_k chunks using dense vector similarity."""
        total = len(self._corpus_docs)
        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for rank, (doc_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": dict(meta) if meta else {},
                    "rank": rank,
                    "distance": dist,
                    "score": round(1.0 / (1.0 + max(float(dist or 0.0), 0.0)), 4),
                }
            )
        return hits

    def bm25_search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Retrieve top_k chunks using BM25 lexical matching."""
        if not self._bm25 or not self._corpus_docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: min(top_k, len(scores))]

        hits: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            hits.append(
                {
                    "id": self._corpus_ids[idx],
                    "text": self._corpus_docs[idx],
                    "metadata": self._corpus_meta[idx],
                    "rank": rank,
                    "score": round(float(scores[idx]), 4),
                }
            )
        return hits

    def hybrid_search(
        self, query: str, top_k: int = 4, rrf_k: int = 60
    ) -> list[dict[str, Any]]:
        """Fuse semantic and BM25 rankings via Reciprocal Rank Fusion (RRF)."""
        candidate_k = max(top_k * 2, 6)
        sem_hits = self.semantic_search(query, top_k=candidate_k)
        bm25_hits = self.bm25_search(query, top_k=candidate_k)

        fused: dict[str, dict[str, Any]] = {}

        for hit in sem_hits:
            doc_id = hit["id"]
            fused.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                },
            )
            fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + hit["rank"])
            fused[doc_id]["semantic_rank"] = hit["rank"]

        for hit in bm25_hits:
            doc_id = hit["id"]
            fused.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "bm25_rank": None,
                },
            )
            fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + hit["rank"])
            fused[doc_id]["bm25_rank"] = hit["rank"]

        ranked = sorted(
            fused.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]

        for item in ranked:
            item["rrf_score"] = round(item["rrf_score"], 5)

        return ranked
