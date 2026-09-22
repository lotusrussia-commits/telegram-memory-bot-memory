from collections import defaultdict, deque
from pathlib import Path

import chromadb


# ============================================================
# КРАТКОСРОЧНАЯ ПАМЯТЬ
# ============================================================

MAX_HISTORY_MESSAGES = 10

memory: dict[int, deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)


def add_message(
    user_id: int,
    role: str,
    content: str,
) -> None:
    """Добавляет сообщение в краткосрочную память."""
    memory[user_id].append(
        {
            "role": role,
            "content": content,
        }
    )


def get_history(
    user_id: int,
) -> list[dict[str, str]]:
    """Возвращает историю текущего диалога."""
    return list(memory[user_id])


def clear_history(user_id: int) -> None:
    """Очищает краткосрочную память пользователя."""
    memory[user_id].clear()


# ============================================================
# CHROMADB
# ============================================================

CHROMA_DIR = Path("data/chroma")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# ПАМЯТЬ ДОКУМЕНТОВ
# ============================================================

documents_collection = chroma_client.get_or_create_collection(
    name="documents"
)


def save_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
) -> None:
    """Сохраняет фрагменты загруженного документа."""
    if not chunks:
        return

    ids = [
        f"{source}-{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source,
        }
        for _ in chunks
    ]

    documents_collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search_memory(
    embedding: list[float],
    top_k: int = 3,
) -> list[str]:
    """Ищет релевантные фрагменты документов."""
    if documents_collection.count() == 0:
        return []

    result = documents_collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )

    documents = result.get("documents", [[]])[0]

    return documents


# ============================================================
# ДОЛГОСРОЧНАЯ ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ
# ============================================================

user_memory_collection = chroma_client.get_or_create_collection(
    name="user_memory"
)


def save_user_memory(
    user_id: int,
    content: str,
    embedding: list[float],
) -> None:
    """Сохраняет информацию в долгосрочную память пользователя."""

    existing_count = user_memory_collection.count()

    memory_id = (
        f"{user_id}-{existing_count}-{abs(hash(content))}"
    )

    user_memory_collection.upsert(
        ids=[memory_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[
            {
                "user_id": str(user_id),
            }
        ],
    )


def search_user_memory(
    user_id: int,
    embedding: list[float],
    top_k: int = 5,
) -> list[str]:
    """Ищет релевантную информацию в памяти пользователя."""

    if user_memory_collection.count() == 0:
        return []

    result = user_memory_collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where={
            "user_id": str(user_id),
        },
    )

    documents = result.get("documents", [[]])[0]

    return documents


def clear_user_memory(user_id: int) -> None:
    """Удаляет долгосрочную память конкретного пользователя."""

    if user_memory_collection.count() == 0:
        return

    user_memory_collection.delete(
        where={
            "user_id": str(user_id),
        }
    )


def clear_all_memory(user_id: int) -> None:
    """Очищает краткосрочную и долгосрочную память."""

    clear_history(user_id)
    clear_user_memory(user_id)