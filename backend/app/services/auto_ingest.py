import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from pinecone import Pinecone

from backend.app.utils.universal_preprocess import process_file_to_chunks
from backend.app.utils.gdrive_service import list_files_in_folder, download_file
from backend.app.utils.state_store import load_state, save_state

# -------------------------------
# 1) Settings & Clients
# -------------------------------
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

RAW_DIR = Path("backend/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

EMBED_BATCH = 64     # chunks per embeddings request
UPSERT_BATCH = 100   # vectors per Pinecone request

_oai = OpenAI(api_key=OPENAI_API_KEY)
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)


# -------------------------------
# 2) Embeddings, in batches
# -------------------------------
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed many chunks per request. Same price as one call per chunk, far fewer round trips."""
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = _oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts[i:i + EMBED_BATCH])
        out.extend(d.embedding for d in resp.data)
    return out


# -------------------------------
# 3) Pinecone writes and deletes
# -------------------------------
def _delete_ids(ids: List[str]) -> None:
    for i in range(0, len(ids), 500):
        _index.delete(ids=ids[i:i + 500])


def upsert_chunks(chunks: List[Dict], file_id: str, name: str) -> List[str]:
    """Store a file's chunks under predictable ids (<file_id>#<n>), so they can be removed later."""
    ids = [f"{file_id}#{n}" for n in range(len(chunks))]
    vectors = embed_batch([ch["text"] for ch in chunks])
    records = [
        (vid, vec, {"title": ch.get("title"), "source": name, "text": ch.get("text"), "file_id": file_id})
        for vid, vec, ch in zip(ids, vectors, chunks)
    ]
    for i in range(0, len(records), UPSERT_BATCH):
        _index.upsert(vectors=records[i:i + UPSERT_BATCH])
    return ids


# -------------------------------
# 4) State helpers
# -------------------------------
def _entry(value) -> Dict:
    """Older state files stored only the modified time. Read both shapes."""
    if isinstance(value, dict):
        return value
    return {"mtime": value, "ids": [], "name": ""}


# -------------------------------
# 5) Process one new or changed file
# -------------------------------
def process_single_file(file_id: str, name: str, mime: str, old_ids: List[str]) -> List[str]:
    print(f"[INFO] Processing file: {name}")
    local_path = RAW_DIR / name.replace("/", "_").strip()
    downloaded_path = download_file(file_id, name, mime, str(local_path))
    chunks = process_file_to_chunks(downloaded_path)

    # Remove the previous version first, so an edited file never leaves stale chunks behind.
    if old_ids:
        _delete_ids(old_ids)

    if not chunks:
        print(f"[SKIP] {name}: no text found")
        return []

    ids = upsert_chunks(chunks, file_id, name)
    print(f"[OK] {name}: {len(ids)} chunks stored")
    return ids


# -------------------------------
# 6) Main pipeline: make the index match the Drive folder
# -------------------------------
def process_new_drive_files() -> Dict:
    """
    Adds new files, replaces edited files, and removes files that were deleted from the Drive folder.
    The Drive folder is the source of truth for the knowledge base.
    """
    if not GOOGLE_DRIVE_FOLDER_ID:
        print("[WARN] GOOGLE_DRIVE_FOLDER_ID not set. Skipping ingest.")
        return {"processed": 0, "skipped": 0, "removed": 0}

    state = {fid: _entry(v) for fid, v in load_state().items()}
    files = list_files_in_folder(GOOGLE_DRIVE_FOLDER_ID)
    in_drive = {f["id"] for f in files}

    processed = skipped = removed = failed = 0

    # --- files deleted from Drive ---
    gone = [fid for fid in state if fid not in in_drive]
    if gone and not files:
        # An empty listing more likely means a permissions or network problem than a deliberately emptied folder.
        print("[WARN] Drive folder returned no files. Not removing anything from the index.")
    else:
        for fid in gone:
            ids = state[fid].get("ids") or []
            if ids:
                _delete_ids(ids)
            print(f"[REMOVED] {state[fid].get('name') or fid}: {len(ids)} chunks deleted")
            del state[fid]
            removed += 1

    # --- new or edited files ---
    for f in files:
        fid, mtime = f["id"], f["modifiedTime"]
        if fid in state and state[fid].get("mtime") == mtime:
            continue
        try:
            old_ids = state.get(fid, {}).get("ids") or []
            ids = process_single_file(fid, f["name"], f["mimeType"], old_ids)
        except Exception as e:  # one bad file must not stop the rest
            print(f"[ERROR] {f['name']}: {type(e).__name__}: {e}")
            failed += 1
            continue
        state[fid] = {"mtime": mtime, "ids": ids, "name": f["name"]}
        if ids:
            processed += 1
        else:
            skipped += 1

    save_state(state)
    return {"processed": processed, "skipped": skipped, "removed": removed, "failed": failed}
