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

# --- Improved system prompt for better answers ---
_SYSTEM = """
You are an AI assistant with access to a knowledge base.
Use ONLY the information from the provided context to answer the user's question.

Follow these rules:
1. Be accurate. If the context doesn't have the answer, clearly say:
   "I don't have enough information in the knowledge base to answer that."
2. Write in a natural, friendly tone — no robotic or overly technical wording.
3. Keep answers concise but clear. Use short paragraphs or bullet points if needed.
4. If sources are used, cite them briefly by title in parentheses — no raw file paths.
5. Never invent information outside the context.
"""

# --- Format context for the LLM ---
def _build_context(matches: List[Dict[str, Any]]) -> str:
    parts = []
    for m in matches:
        md = m.get("metadata", {}) or {}
        title = md.get("title") or md.get("source") or "Untitled"
        title = title.replace("_", " ").replace(".txt", "").replace(".pdf", "")
        text = md.get("text") or ""
        parts.append(f"[{title}]\n{text}")
    return "\n\n---\n\n".join(parts)

# --- Extract top N sources for display ---
def _get_top_sources(matches: List[Dict[str, Any]], n: int = 3) -> List[str]:
    sources = []
    for m in matches[:n]:
        title = m.get("metadata", {}).get("title") or m.get("metadata", {}).get("source") or "Untitled"
        title = title.replace("_", " ").replace(".txt", "").replace(".pdf", "")
        sources.append(f"📄 {title}")
    return sources

# --- LLM call ---
def answer_from_context(query: str, matches: List[Dict[str, Any]]) -> str:
    context = _build_context(matches)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]

    resp = _oai.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

# --- Main pipeline for end user ---
def rag_answer(query: str):
    matches = retrieve(query)
    answer = answer_from_context(query, matches)

    top_sources = []
    for m in matches[:3]:
        md = m.get("metadata", {}) or {}
        title = md.get("title") or md.get("source") or "Untitled"
        title = title.replace("_", " ").replace(".txt", "")
        top_sources.append(f"📄 {title}")

    return {
        "question": query,
        "answer": answer,
        "sources": top_sources,
        "matches": matches   # ✅ put it back so routes.py works
    }



