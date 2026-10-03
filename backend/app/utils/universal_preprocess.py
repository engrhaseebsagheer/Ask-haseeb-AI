import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader
from markdown import markdown

# -------------------------------
# 1) Directory Paths
# -------------------------------
BASE_DIR = Path("backend/data")
RAW_DIR = BASE_DIR / "raw"
INTERIM_DIR = BASE_DIR / "interim"
CHUNK_DIR = BASE_DIR / "processed/chunks"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 2) Loaders for different file types
# -------------------------------
def load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_md(path: Path) -> str:
    md = load_text(path)
    html = markdown(md, output_format="html")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def load_html(path: Path) -> str:
    html = load_text(path)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator="\n")

def load_any(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".md":
        return load_md(path)
    if suffix in {".html", ".htm"}:
        return load_html(path)
    if suffix == ".txt":
        return load_text(path)
    return load_text(path)  # fallback for unknown types

# -------------------------------
# 3) Cleaning text
# -------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"https?://\S+", line.strip()):
            continue
        if line.strip().lower() in {"share", "login", "sign in", "subscribe"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()

# -------------------------------
# 4) Chunking (plain Python, no extra packages)
# -------------------------------
SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_text(text: str, size: int = 1000, overlap: int = 200, seps=SEPARATORS) -> List[str]:
    """Split text into pieces of at most `size` characters, breaking at paragraphs first,
    then lines, sentences and words. Each piece starts with the tail of the previous one,
    so meaning carries across the boundary."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    sep = next((s for s in seps if s in text), None)
    if sep is None:  # one unbroken run of characters: cut it
        step = size - overlap
        return [text[i:i + size] for i in range(0, len(text), step)]

    rest = seps[seps.index(sep) + 1:]
    pieces: List[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        pieces.extend(split_text(part, size, overlap, rest) if len(part) > size else [part])

    chunks: List[str] = []
    current = ""
    joiner = sep if sep.strip() == "" else sep.strip() + " "
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:]
            cut = tail.find(" ")
            tail = tail[cut + 1:] if cut != -1 else tail  # start the overlap on a word boundary
            current = f"{tail}{joiner}{piece}" if len(tail) + len(joiner) + len(piece) <= size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    source: str,
    title: str,
    chunk_tokens: int = 350,
    overlap: int = 50,
) -> List[Dict]:
    """Split text into chunks with metadata. 1,000 characters is roughly 250 tokens,
    which keeps every chunk under the 350-token target without a tokenizer."""
    return [
        {"id": str(uuid.uuid4()), "text": ch, "source": source, "title": title, "tokens": len(ch) // 4}
        for ch in split_text(text, size=1000, overlap=200)
    ]

# -------------------------------
# 5) Iterator for batch processing
# -------------------------------
def iter_files() -> Iterable[Path]:
    for f in RAW_DIR.rglob("*"):
        if f.is_file():
            print(f"[DEBUG] Found: {f}")
            yield f

# -------------------------------
# 6) Reusable function for single file
# -------------------------------
def process_file_to_chunks(file_path: str, chunk_tokens: int = 350, overlap: int = 50) -> List[Dict]:
    """
    Process a single file: load, clean, chunk, and return chunks as a list of dicts.
    Saves interim text and per-file chunk JSONL for consistency.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        print(f"[SKIP] {file_path}: File not found")
        return []

    raw_text = load_any(path)
    if not raw_text.strip():
        print(f"[SKIP] {file_path}: No extractable text")
        return []

    cleaned = clean_text(raw_text)

    # Save interim text for reference/debugging
    interim_path = INTERIM_DIR / f"{path.stem}.txt"
    interim_path.write_text(cleaned, encoding="utf-8")

    title = path.stem
    chunks = chunk_text(cleaned, source=str(path), title=title, chunk_tokens=chunk_tokens, overlap=overlap)

    # Save per-file chunks JSONL
    out_file = CHUNK_DIR / f"{path.stem}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    return chunks

# -------------------------------
# 7) Batch processing entry point
# -------------------------------
def main():
    all_chunks = []
    file_count = 0

    print(f"\n[INFO] Starting preprocessing from: {RAW_DIR}\n")

    for path in iter_files():
        file_count += 1
        print(f"[LOAD] {path.name}")
        chunks = process_file_to_chunks(str(path))
        all_chunks.extend(chunks)
        print(f"[OK] {path.name}: {len(chunks)} chunks")

    # Save all chunks in one aggregate file
    agg_file = CHUNK_DIR / "all_chunks.jsonl"
    with agg_file.open("w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Processed {file_count} files")
    print(f"[RESULT] Total chunks: {len(all_chunks)}")
    print(f"[OUT] Per-file: {CHUNK_DIR}/*.jsonl")
    print(f"[OUT] Aggregate: {agg_file}")

if __name__ == "__main__":
    main()
