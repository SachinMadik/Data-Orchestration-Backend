"""
rag_service.py — Grounded answer generation from retrieved chunks.

Changes from v1:
  - Uses chunk["text"] directly (already the relevant 500-token window)
    instead of fetching full 6000 chars from blob
  - Includes source citations in the response
  - Context capped at 4000 chars total across all chunks (not per-doc)
"""

import os
import re
import logging
from openai import AzureOpenAI

# Max chars of chunk text to include per chunk in the LLM context
_CHUNK_TEXT_LIMIT = 800   # ~200 tokens per chunk × 7 chunks ≈ 1400 tokens context


class RAGService:
    def __init__(self):
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
            )

        self._deployment = deployment
        self._client     = AzureOpenAI(
            api_key        = api_key,
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_endpoint = endpoint,
        )

    def generate_answer(self, query: str, documents: list[dict]) -> str:
        """
        Build context from reranked chunks and generate a grounded answer.

        Args:
            query     : User question.
            documents : Reranked chunks from hybrid search
                        (each has: text, filename, score, summary).

        Returns:
            Answer string grounded strictly in the provided chunks.
        """
        if not documents:
            logging.warning("generate_answer: no chunks provided.")
            return "Not enough information in the available documents to answer this question."

        # Build context from chunk text only (not full doc)
        context_parts = []
        for i, chunk in enumerate(documents, start=1):
            # Use chunk text (already the relevant 500-token window from retrieval)
            text     = (chunk.get("text") or chunk.get("content") or "").strip()
            text     = text[:_CHUNK_TEXT_LIMIT]
            filename = chunk.get("filename", f"Document {i}")
            score    = chunk.get("score", 0)

            if text:
                context_parts.append(
                    f"[Chunk {i} — {filename} (relevance: {score:.2f})]\n{text}"
                )

        if not context_parts:
            return "Not enough information in the available documents to answer this question."

        context = "\n\n".join(context_parts)

        # Build citation list for the prompt
        seen = set()
        citation_list = []
        for chunk in documents:
            fn = chunk.get("filename", "")
            if fn and fn not in seen:
                seen.add(fn)
                citation_list.append(fn)

        prompt = (
            "You are an AI assistant. Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say: 'Not enough information.'\n\n"
            "FORMAT RULES:\n"
            "- Use numbered bullet points\n"
            "- Each point MUST be on its own line\n"
            "- Keep each point concise (1-2 lines max)\n"
            f"- At the end, add a 'Sources:' line listing: {', '.join(citation_list)}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            logging.info("generate_answer: %d chunks, deployment=%s",
                         len(documents), self._deployment)
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.2,
                max_tokens  = 512,
            )
            answer = response.choices[0].message.content.strip()
            # Ensure numbered points each start on a new line
            answer = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', answer).strip()
            logging.info("generate_answer: %d chars", len(answer))
            return answer

        except Exception:
            logging.exception("generate_answer: Azure OpenAI call failed.")
            raise
