from pathlib import Path

from docx import Document
from pypdf import PdfReader

from app.llm import create_embedding
from app.memory import save_chunks


UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )


def read_docx(path: Path) -> str:
    document = Document(str(path))

    return "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return read_txt(path)

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix == ".docx":
        return read_docx(path)

    raise ValueError(
        "Поддерживаются только TXT, PDF и DOCX."
    )


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


async def index_document(path: Path) -> int:
    text = load_document(path)
    chunks = split_text(text)

    if not chunks:
        return 0

    embeddings = []

    for chunk in chunks:
        embedding = await create_embedding(chunk)
        embeddings.append(embedding)

    save_chunks(
        chunks=chunks,
        embeddings=embeddings,
        source=path.name,
    )

    return len(chunks)