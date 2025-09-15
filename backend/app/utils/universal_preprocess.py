import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader
from markdown import markdown
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
# 2) Loaders
# -------------------------------
def load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
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
# 3) Cleaning
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
# 4) Chunking with LangChain
# -------------------------------
def chunk_text(
    text: str,
    source: str,
    title: str,
    chunk_tokens: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[Dict]:
    enc = tiktoken.get_encoding(encoding_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    rough_chunks = splitter.split_text(text)

    final_chunks = []
    for ch in rough_chunks:
        tokens = enc.encode(ch)
        if len(tokens) <= chunk_tokens:
            final_chunks.append({
                "id": str(uuid.uuid4()),
                "text": ch.strip(),
                "source": source,
                "title": title,
                "tokens": len(tokens)
            })
        else:
            start = 0
            step = chunk_tokens - overlap
            while start < len(tokens):
                window = tokens[start:start + chunk_tokens]
                piece = enc.decode(window).strip()
                final_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": piece,
                    "source": source,
                    "title": title,
                    "tokens": len(window)
                })
                start += step
    return final_chunks

# -------------------------------
# 5) Main Runner
# -------------------------------
def iter_files() -> Iterable[Path]:
    for f in RAW_DIR.rglob("*"):
        if f.is_file():
            print(f"[DEBUG] Found: {f}")
            yield f

def main():
    all_chunks = []
    file_count = 0

    print(f"\n[INFO] Starting preprocessing from: {RAW_DIR}\n")

    for path in iter_files():
        file_count += 1
        print(f"[LOAD] {path.name}")

        raw_text = load_any(path)
        if not raw_text.strip():
            print(f"[SKIP] {path.name}: No extractable text")
            continue

        cleaned = clean_text(raw_text)

        # Save cleaned text
        interim_path = INTERIM_DIR / f"{path.stem}.txt"
        interim_path.write_text(cleaned, encoding="utf-8")

        # Use filename as title
        title = path.stem

        # Chunk the cleaned text with title
        chunks = chunk_text(cleaned, source=str(path), title=title)
        all_chunks.extend(chunks)

        # Save per-file chunks
        out_file = CHUNK_DIR / f"{path.stem}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        print(f"[OK] {path.name}: {len(chunks)} chunks")

    # Save all chunks in one file
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
