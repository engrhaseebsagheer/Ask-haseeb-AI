from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone
from ..utils.config import get_settings

settings = get_settings()

# --- Clients ---
_oai = OpenAI(api_key=settings.OPENAI_API_KEY)
_pc = Pinecone(api_key=settings.PINECONE_API_KEY)
_index = _pc.Index(settings.PINECONE_INDEX_NAME)

# --- Embeddings ---
def _embed(text: str) -> List[float]:
    emb = _oai.embeddings.create(
        model=settings.OPENAI_EMBED_MODEL,
        input=text
    )
    return emb.data[0].embedding

# --- Retrieval from Pinecone ---
def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    k = top_k or settings.TOP_K
    qvec = _embed(query)

    res = _index.query(
        vector=qvec,
        top_k=k,
        include_metadata=True
    )

    matches = res.get("matches", []) or []
    if settings.MIN_SCORE > 0:
        matches = [m for m in matches if float(m.get("score", 0)) >= settings.MIN_SCORE]
    return matches

# --- Compose system prompt for grounded answers ---
_SYSTEM = (
    "You are a precise assistant for Haseeb’s RAG system. "
    "Answer strictly from the provided context. If the answer is not in context, say "
    "\"I don't have enough information in the knowledge base to answer that.\" "
    "Cite brief file titles inline in parentheses when helpful."
)

def _build_context(matches: List[Dict[str, Any]]) -> str:
    parts = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        title = md.get("title") or md.get("source") or "Untitled"
        text = md.get("text") or ""
        parts.append(f"[{title}]\n{text}")
    return "\n\n---\n\n".join(parts)

# --- LLM call ---
def answer_from_context(query: str, matches: List[Dict[str, Any]]) -> str:
    context = _build_context(matches)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer clearly and concisely."}
    ]

    resp = _oai.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

# --- Main pipeline ---
def rag_answer(query: str) -> tuple[str, List[Dict[str, Any]]]:
    matches = retrieve(query)
    answer = answer_from_context(query, matches)
    return answer, matches
