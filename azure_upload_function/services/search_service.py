"""
search_service.py — Hybrid BM25 + vector search with cosine reranking.

Architecture (Free tier compatible):
  1. BM25 full-text search via Azure AI Search  → top-20 candidates
  2. Cosine similarity reranking on embeddings  → reorder by semantic score
  3. Return top-K after reranking

This gives hybrid retrieval without requiring Basic/Standard tier vector fields.
Embeddings are stored in Azure Table Storage alongside each chunk.
"""

import os
import math
import logging
import requests
from time import sleep
from services.config import require_env

SEARCH_INDEX = "documents-index"
_API_VERSION = "2023-11-01"

_headers: dict | None = None


def _get_headers() -> dict:
    global _headers
    if _headers is None:
        _headers = {
            "Content-Type": "application/json",
            "api-key":      require_env("AZURE_SEARCH_KEY"),
        }
    return _headers


def _endpoint() -> str:
    return require_env("AZURE_SEARCH_ENDPOINT").rstrip("/")


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def delete_index() -> bool:
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}?api-version={_API_VERSION}"
    resp = requests.delete(url, headers=_get_headers(), timeout=10)
    if resp.status_code in (200, 204, 404):
        logging.info("AI Search index '%s' deleted.", SEARCH_INDEX)
        return True
    logging.error("Failed to delete index: %s — %s", resp.status_code, resp.text)
    return False


def ensure_index() -> None:
    """
    Create/verify the BM25 keyword index.
    chunk_id is the key (one record per chunk, not per document).
    embedding stored in Table Storage — not in Search index (free tier).
    """
    url   = f"{_endpoint()}/indexes/{SEARCH_INDEX}?api-version={_API_VERSION}"
    check = requests.get(url, headers=_get_headers(), timeout=10)
    if check.status_code == 200:
        logging.info("AI Search index '%s' already exists.", SEARCH_INDEX)
        return

    schema = {
        "name": SEARCH_INDEX,
        "fields": [
            {"name": "id",          "type": "Edm.String",             "key": True,  "searchable": False, "filterable": True,  "retrievable": True},
            {"name": "doc_id",      "type": "Edm.String",             "key": False, "searchable": False, "filterable": True,  "retrievable": True},
            {"name": "filename",    "type": "Edm.String",             "key": False, "searchable": True,  "filterable": True,  "retrievable": True},
            {"name": "chunk_index", "type": "Edm.Int32",              "key": False, "searchable": False, "filterable": True,  "retrievable": True},
            {"name": "content",     "type": "Edm.String",             "key": False, "searchable": True,  "filterable": False, "retrievable": True},
            {"name": "summary",     "type": "Edm.String",             "key": False, "searchable": True,  "filterable": False, "retrievable": True},
            {"name": "tags",        "type": "Collection(Edm.String)", "key": False, "searchable": True,  "filterable": True,  "retrievable": True},
            {"name": "blob_url",    "type": "Edm.String",             "key": False, "searchable": False, "filterable": False, "retrievable": True},
        ],
    }

    resp = requests.put(url, headers=_get_headers(), json=schema, timeout=20)
    if resp.status_code not in (200, 201):
        logging.error("Index creation failed: %s", resp.text)
        resp.raise_for_status()
    logging.info("AI Search index '%s' created.", SEARCH_INDEX)


# ---------------------------------------------------------------------------
# Index a chunk (called once per chunk during upload)
# ---------------------------------------------------------------------------

def index_document(
    doc_id:      str,
    filename:    str,
    content:     str,
    summary:     str,
    tags:        list[str],
    blob_url:    str,
    embedding:   list[float] = None,  # stored in Table Storage, not here
    chunk_index: int = 0,
    chunk_id:    str = "",
    retries:     int = 3,
) -> None:
    """Index one chunk into Azure AI Search (BM25 fields only)."""
    record_id = chunk_id or doc_id
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}/docs/index?api-version={_API_VERSION}"
    body = {
        "value": [{
            "@search.action": "upload",
            "id":          record_id,
            "doc_id":      doc_id,
            "filename":    filename,
            "chunk_index": chunk_index,
            "content":     content[:32000],
            "summary":     summary,
            "tags":        tags,
            "blob_url":    blob_url,
        }]
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=_get_headers(), json=body, timeout=20)
            if resp.status_code in (200, 207):
                logging.info("Indexed chunk: %s (chunk %d)", filename, chunk_index)
                return
            logging.warning("index_document attempt %d/%d: %s", attempt, retries, resp.status_code)
        except requests.RequestException as exc:
            logging.warning("index_document attempt %d/%d: %s", attempt, retries, exc)
        if attempt < retries:
            sleep(2 ** attempt)

    raise RuntimeError(f"AI Search indexing failed for {record_id} after {retries} attempts")


# ---------------------------------------------------------------------------
# Hybrid search: BM25 candidates → cosine rerank
# ---------------------------------------------------------------------------

def vector_search(
    query_embedding: list[float],
    query_text:      str,
    top:             int = 7,
    filename_filter: str = "",
    min_score:       float = 0.01,
) -> list[dict]:
    """
    Hybrid retrieval:
      1. BM25 search → fetch top-20 candidates from Azure AI Search
      2. Cosine rerank using query_embedding vs stored chunk embeddings
      3. Return top-K reranked results

    Falls back to BM25-only ranking if embeddings unavailable.
    """
    bm25_candidates = _bm25_search(query_text, top=20,
                                   filename_filter=filename_filter,
                                   min_score=min_score)

    if not bm25_candidates:
        return []

    # Rerank with cosine similarity if query embedding available
    if query_embedding:
        reranked = _cosine_rerank(bm25_candidates, query_embedding, top_k=top)
        logging.info("hybrid_search: %d BM25 candidates → %d after rerank",
                     len(bm25_candidates), len(reranked))
        return reranked

    # Fallback: return BM25 top-K
    return bm25_candidates[:top]


# ---------------------------------------------------------------------------
# BM25 full-text search
# ---------------------------------------------------------------------------

def _bm25_search(
    query_text:      str,
    top:             int = 20,
    filename_filter: str = "",
    min_score:       float = 0.01,
) -> list[dict]:
    url  = f"{_endpoint()}/indexes/{SEARCH_INDEX}/docs/search?api-version={_API_VERSION}"
    body: dict = {
        "search":       query_text,
        "searchFields": "content,summary,tags,filename",
        "select":       "id,doc_id,filename,chunk_index,summary,content,blob_url,tags",
        "top":          top,
        "queryType":    "simple",
    }
    if filename_filter:
        body["filter"] = f"search.ismatch('{filename_filter}', 'filename')"

    try:
        resp = requests.post(url, headers=_get_headers(), json=body, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("value", [])
        return [
            {
                "id":          r.get("id", ""),
                "doc_id":      r.get("doc_id", r.get("id", "")),
                "filename":    r.get("filename", ""),
                "chunk_index": r.get("chunk_index", 0),
                "blob_url":    r.get("blob_url", ""),
                "summary":     r.get("summary", ""),
                "tags":        r.get("tags", []),
                "text":        r.get("content", ""),
                "bm25_score":  round(r.get("@search.score", 0), 4),
                "score":       round(r.get("@search.score", 0), 4),
                "embedding":   [],   # populated by reranker from Table Storage
            }
            for r in results
            if r.get("@search.score", 0) >= min_score
        ]
    except requests.HTTPError as exc:
        logging.error("BM25 search error: %s — %s",
                      exc.response.status_code, exc.response.text[:200])
        return []
    except Exception:
        logging.exception("BM25 search failed.")
        return []


# ---------------------------------------------------------------------------
# Cosine reranker
# ---------------------------------------------------------------------------

def _cosine_rerank(
    candidates:      list[dict],
    query_embedding: list[float],
    top_k:           int = 7,
) -> list[dict]:
    """
    Fetch stored chunk embeddings from Table Storage and rerank candidates
    by cosine similarity to the query embedding.

    Hybrid score = 0.4 * bm25_norm + 0.6 * cosine_sim
    """
    try:
        from services.table_service import TableService
        table_svc = TableService()

        # Fetch embeddings for all candidates in one pass (fast point lookups)
        chunk_ids = [c["id"] for c in candidates]
        doc_ids   = {c["id"]: c.get("doc_id", "") for c in candidates}
        embeddings_map = table_svc.get_chunk_embeddings(chunk_ids, doc_ids)

        scored = []
        bm25_scores = [c["bm25_score"] for c in candidates]
        bm25_max    = max(bm25_scores) if bm25_scores else 1.0

        for chunk in candidates:
            emb = embeddings_map.get(chunk["id"], [])
            if emb and query_embedding:
                cos_sim = _cosine(query_embedding, emb)
            else:
                cos_sim = 0.0

            bm25_norm    = chunk["bm25_score"] / bm25_max if bm25_max > 0 else 0
            hybrid_score = 0.4 * bm25_norm + 0.6 * cos_sim

            scored.append({**chunk, "score": round(hybrid_score, 4), "cosine": round(cos_sim, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        logging.info("reranker: top scores = %s",
                     [round(s["score"], 3) for s in scored[:top_k]])
        return scored[:top_k]

    except Exception:
        logging.exception("cosine_rerank failed — falling back to BM25 order.")
        return candidates[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
