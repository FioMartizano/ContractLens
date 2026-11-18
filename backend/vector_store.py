import json
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

import chromadb


_DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "chroma_db"
_PERSIST_DIRECTORY = Path(
    os.getenv("CHROMA_PERSIST_DIRECTORY", str(_DEFAULT_DIRECTORY))
).expanduser()
_COLLECTION_NAME = "contract_documents"
_STANDARD_COLLECTION_NAME = "standard_clauses"
_ANALYSIS_COLLECTION_NAME = "contract_analyses"
_client = None
_collections = {}
_collection_lock = RLock()


def _get_collection(name: str = _COLLECTION_NAME):
    global _client

    with _collection_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(_PERSIST_DIRECTORY))

        if name not in _collections:
            _collections[name] = _client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        return _collections[name]


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks


def index_document(
    filename: str,
    file_type: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(uuid.uuid4())
    chunk_texts = []
    metadatas = []
    ids = []

    for page in pages:
        for chunk_index, chunk in enumerate(_chunk_text(page["text"])):
            chunk_texts.append(chunk)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                }
            )
            ids.append(f"{document_id}:{page['page']}:{chunk_index}")

    if not chunk_texts:
        raise ValueError("No readable text was found in the document.")

    _get_collection().add(
        ids=ids,
        documents=chunk_texts,
        metadatas=metadatas,
    )

    return {"document_id": document_id, "chunks_indexed": len(chunk_texts)}


def ensure_standard_clauses(clauses: list[dict[str, str]]) -> None:
    collection = _get_collection(_STANDARD_COLLECTION_NAME)
    ids = [clause["id"] for clause in clauses]
    existing = collection.get(ids=ids, include=["metadatas"])
    existing_ids = set(existing["ids"])
    missing = [clause for clause in clauses if clause["id"] not in existing_ids]
    if not missing:
        return

    collection.upsert(
        ids=[clause["id"] for clause in missing],
        documents=[clause["text"] for clause in missing],
        metadatas=[
            {"title": clause["title"], "clause_type": clause["clause_type"]}
            for clause in missing
        ],
    )


def retrieve_standard_clauses(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    collection = _get_collection(_STANDARD_COLLECTION_NAME)
    if collection.count() == 0:
        return []

    matches = collection.query(query_texts=[query], n_results=top_k)
    return [
        {
            "id": clause_id,
            "title": metadata["title"],
            "clause_type": metadata["clause_type"],
            "text": text,
            "distance": distance,
        }
        for clause_id, text, metadata, distance in zip(
            matches["ids"][0],
            matches["documents"][0],
            matches["metadatas"][0],
            matches["distances"][0],
        )
    ]


def save_analysis(document_id: str, filename: str, analysis: dict[str, Any]) -> None:
    _get_collection(_ANALYSIS_COLLECTION_NAME).upsert(
        ids=[document_id],
        documents=[json.dumps(analysis)],
        metadatas=[{"document_id": document_id, "filename": filename}],
    )


def get_analysis(document_id: str) -> dict[str, Any] | None:
    record = _get_collection(_ANALYSIS_COLLECTION_NAME).get(
        ids=[document_id],
        include=["documents"],
    )
    if not record["documents"]:
        return None
    return json.loads(record["documents"][0])


def list_analyses() -> list[dict[str, Any]]:
    records = _get_collection(_ANALYSIS_COLLECTION_NAME).get(
        include=["documents"],
    )
    return sorted(
        (json.loads(document) for document in records["documents"] if document),
        key=lambda analysis: analysis.get("uploaded_at", ""),
        reverse=True,
    )


def search_documents(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_options = {
        "query_texts": [query],
        "n_results": top_k,
    }
    if document_id:
        query_options["where"] = {"document_id": document_id}

    matches = collection.query(**query_options)
    documents = matches["documents"][0]
    metadatas = matches["metadatas"][0]
    distances = matches["distances"][0]

    return [
        {
            "text": text,
            "metadata": metadata,
            "distance": distance,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
