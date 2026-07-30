import asyncio
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

from pypdf import PdfReader
from dotenv import load_dotenv

from app import config
from app.ingestion.embedder import embed_batch
from app.ingestion.utils.chunking_methods import chunk_text
from app.ingestion.utils.faiss import FAISSStore
from app.logger import Logger

logger = Logger.get_logger(__name__)
load_dotenv()


def load_pdf_pages(file_paths: Iterable[str]) -> List[Dict[str, Any]]:
    """Load non-empty PDF pages as parent documents."""
    documents = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        reader = PdfReader(file_path)

        for page_index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            parent_id = f"{filename}:page:{page_index + 1}"
            documents.append(
                {
                    "id": parent_id,
                    "text": text,
                    "metadata": {
                        "source": file_path,
                        "filename": filename,
                        "page": page_index + 1,
                        "parent_id": parent_id,
                        "document_type": "page",
                    },
                }
            )

    return documents


async def _summarize_document(client, document, semaphore):
    """Summarize one parent document while respecting the concurrency limit."""
    prompt = f"""Summarize this legal document page for retrieval.

Preserve the important parties, dates, statutes, legal issues, holdings,
procedural history, and outcomes. Use concise factual language. Do not add
information that is not present in the page.

PAGE:
{document["text"][:config.SUMMARY_MAX_INPUT_CHARS]}
"""

    async with semaphore:
        response = await client.responses.create(
            model=config.SUMMARY_MODEL,
            input=prompt,
        )

    return {
        "id": f'{document["id"]}:summary',
        "text": response.output_text.strip(),
        "metadata": {
            **document["metadata"],
            "document_type": "summary",
        },
    }


async def summarize_documents(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize parent documents concurrently with the OpenAI Responses API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(config.SUMMARY_CONCURRENCY)
    tasks = [
        _summarize_document(client, document, semaphore)
        for document in documents
    ]
    return await asyncio.gather(*tasks)


def create_detailed_chunks(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Split every parent document and retain its parent relationship."""
    detailed_chunks = []

    for document in documents:
        chunks = chunk_text(
            document["text"],
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        for chunk_index, chunk in enumerate(chunks):
            detailed_chunks.append(
                {
                    "id": f'{document["id"]}:chunk:{chunk_index}',
                    "text": chunk,
                    "metadata": {
                        **document["metadata"],
                        "chunk_id": chunk_index,
                        "document_type": "detail",
                    },
                }
            )

    return detailed_chunks


def create_bm25_index(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build serializable BM25 statistics for the detailed chunks."""
    tokenized_documents = [
        re.findall(r"\w+", document.get("text", "").lower())
        for document in documents
    ]
    term_frequencies = [dict(Counter(tokens)) for tokens in tokenized_documents]
    document_frequencies = Counter(
        term
        for tokens in tokenized_documents
        for term in set(tokens)
    )
    document_lengths = [len(tokens) for tokens in tokenized_documents]

    return {
        "document_count": len(documents),
        "average_document_length": (
            sum(document_lengths) / len(documents) if documents else 0
        ),
        "document_lengths": document_lengths,
        "term_frequencies": term_frequencies,
        "document_frequencies": dict(document_frequencies),
    }


def _create_vector_store(
    documents: List[Dict[str, Any]],
    index_path: str,
    metadata_path: str,
) -> FAISSStore:
    embeddings = embed_batch([document["text"] for document in documents])
    store = FAISSStore(len(embeddings[0]))
    store.add(embeddings, documents)
    store.save(index_path=index_path, metadata_path=metadata_path)
    return store


def ingest_hierarchical_pdfs(file_paths: Iterable[str]) -> Dict[str, Any]:
    """Create and persist summary-level and detail-level retrieval indices."""
    file_paths = list(file_paths)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is required to create document summaries"
        )

    parent_documents = load_pdf_pages(file_paths)
    if not parent_documents:
        raise ValueError("No extractable PDF pages were found")

    logger.info("Summarizing %s PDF pages", len(parent_documents))
    summary_documents = asyncio.run(summarize_documents(parent_documents))
    detailed_documents = create_detailed_chunks(parent_documents)
    if not detailed_documents:
        raise ValueError("No detailed chunks were generated")

    os.makedirs("vector_store", exist_ok=True)

    summary_store = _create_vector_store(
        summary_documents,
        config.SUMMARY_VECTOR_STORE_PATH,
        config.SUMMARY_METADATA_PATH,
    )
    detail_store = _create_vector_store(
        detailed_documents,
        config.VECTOR_STORE_PATH,
        config.METADATA_PATH,
    )

    bm25_index = create_bm25_index(detailed_documents)
    with open(config.BM25_INDEX_PATH, "w", encoding="utf-8") as index_file:
        json.dump(bm25_index, index_file)

    logger.info(
        "Created hierarchical indices with %s summaries and %s detailed chunks",
        len(summary_documents),
        len(detailed_documents),
    )
    return {
        "files": len(file_paths),
        "pages": len(parent_documents),
        "summaries": len(summary_store.metadata),
        "chunks": len(detail_store.metadata),
    }
