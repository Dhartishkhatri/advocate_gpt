import json
import os
import re
from collections import Counter
from typing import List, Dict, Any
from app import config
from app.ingestion.embedder import embed_batch
from app.ingestion.utils.faiss import FAISSStore
from app.ingestion.utils.extract_text_from_pdf_methods import (
    extract_text_from_pdf_using_pypdf,
    extract_text_from_pdf_using_unstructured,
    extract_text_from_pdf_using_pdfplumber,
    extract_text_from_pdf_using_pymupdf,
    extract_text_from_pdf_using_tesseract,
    extract_text_from_pdf_using_pdfminer,
)
from app.ingestion.utils.chunking_methods import (
    chunk_text,
    chunk_by_words,
    chunk_by_tokens,
    chunk_by_sentences,
    chunk_by_paragraphs,
    chunk_by_custom_splitter,
    chunk_by_adding_contextual_document_header,
    chunk_by_adding_contextual_chunk_header,
    chunk_by_semantic,
)

pdf_text_extraction_methods = {
    "extract_text_from_pdf_using_pypdf": extract_text_from_pdf_using_pypdf,
    "extract_text_from_pdf_using_unstructured": extract_text_from_pdf_using_unstructured,
    "extract_text_from_pdf_using_pdfplumber": extract_text_from_pdf_using_pdfplumber,
    "extract_text_from_pdf_using_pymupdf": extract_text_from_pdf_using_pymupdf,
    "extract_text_from_pdf_using_tesseract": extract_text_from_pdf_using_tesseract,
    "extract_text_from_pdf_using_pdfminer": extract_text_from_pdf_using_pdfminer,
}

pdf_chunking_methods = {
    "chunk_text": chunk_text,
    "chunk_by_words": chunk_by_words,
    "chunk_by_tokens": chunk_by_tokens,
    "chunk_by_sentences": chunk_by_sentences,
    "chunk_by_paragraphs": chunk_by_paragraphs,
    "chunk_by_custom_splitter": chunk_by_custom_splitter,
    "chunk_by_adding_contextual_chunk_header": chunk_by_adding_contextual_chunk_header,
    "chunk_by_adding_contextual_document_header": chunk_by_adding_contextual_document_header,
    "chunk_by_semantic": chunk_by_semantic,
}

from app.logger import Logger
logger = Logger.get_logger(__name__)

def normalize_chunks(chunks: List[Any], file_path: str) -> List[Dict[str, Any]]:
    """
    Convert chunks into standard document format for FAISS metadata.
    """
    normalized_docs = []

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = str(chunk.get("text", "")).strip()
            metadata = dict(chunk.get("metadata", {}) or {})

            if not metadata:
                metadata = {}
        else:
            text = str(chunk).strip()
            metadata = {}

        if not text:
            continue

        normalized_docs.append({
            "id": f"{os.path.basename(file_path)}_{i}",
            "text": text,
            "metadata": {
                **metadata,
                "source": file_path,
                "filename": os.path.basename(file_path),
                "chunk_id": i,
            }
        })

    return normalized_docs


def chunk_to_embed(docs: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Create embeddings for all chunks.
    """
    texts = [doc["text"] for doc in docs]

    if not texts:
        return []

    embeddings = embed_batch(texts)

    return embeddings


def generate_document_title(document_text: str, document_title_guidance: str = "") -> str:
    """
    Extract a document title using OpenAI.
    """
    from openai import OpenAI
    import tiktoken

    max_content_tokens = 4000
    model_name = "gpt-4o-mini"
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    tokens = encoder.encode(document_text, disallowed_special=())
    truncated_text = encoder.decode(tokens[:max_content_tokens])
    truncation_message = ""

    if len(tokens) >= max_content_tokens:
        truncation_message = (
            "Also note that the document text provided below is just the first "
            "~3000 words of the document. That should be plenty for this task. "
            "Your response should still pertain to the entire document, not just "
            "the text provided below."
        )

    prompt = f"""
INSTRUCTIONS
What is the title of the following document?

Your response MUST be the title of the document, and nothing else. DO NOT respond with anything else.

{document_title_guidance}

{truncation_message}

DOCUMENT
{truncated_text}
""".strip()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_content_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()



def create_bm25_index(documents, index_path=None):
    tokenized_documents = [
        re.findall(r"\w+", document.get("text", "").lower())
        for document in documents
    ]

    term_frequencies = [
        dict(Counter(tokens))
        for tokens in tokenized_documents
    ]

    document_frequencies = Counter(
        term
        for tokens in tokenized_documents
        for term in set(tokens)
    )

    document_lengths = [
        len(tokens)
        for tokens in tokenized_documents
    ]

    index = {
        "document_count": len(documents),
        "average_document_length": (
            sum(document_lengths) / len(documents)
            if documents else 0
        ),
        "document_lengths": document_lengths,
        "term_frequencies": term_frequencies,
        "document_frequencies": dict(document_frequencies),
    }

    return index

def ingest_pdf(file_path: str):
    """
    Full PDF ingestion pipeline:
    1. Extract PDF text
    2. Chunk text
    3. Embed chunks
    4. Store embeddings in FAISS
    5. Save FAISS index
    """
    logger.info(f"Starting ingestion pipeline for: {file_path}")
    try:
        pdf_text_extraction_method = pdf_text_extraction_methods.get(config.PDF_TEXT_EXTRACTION_METHOD)
        text = pdf_text_extraction_method(file_path)

        chunking_method = pdf_chunking_methods.get(config.PDF_CHUNKING_METHOD)
        if config.PDF_CHUNKING_METHOD == "chunk_by_adding_contextual_chunk_header":
            header = generate_document_title(text)
            chunks = chunking_method(text,header=header)
            
        else:
            chunks = chunking_method(text)

        if not chunks:
            raise ValueError("No text chunks were extracted from the PDF")

        all_docs = normalize_chunks(
            chunks=chunks,
            file_path=file_path
        )
        os.makedirs("vector_store", exist_ok=True)
        bm25_index = create_bm25_index(all_docs)
        embeddings = chunk_to_embed(all_docs)

        if len(embeddings) == 0:
            raise ValueError("No embeddings were generated")

        dim = len(embeddings[0])

        store = FAISSStore(dim)

        store.add(
            embeddings=embeddings,
            docs=all_docs
        )
        store.save()

        with open(config.BM25_INDEX_PATH, "w", encoding="utf-8") as index_file:
            json.dump(bm25_index, index_file)

        return store

    except Exception as e:
        logger.error(f"Error during {file_path} ingestion: {str(e)}")
        return None


def process_pdf(path: str):
    """
    Process a single PDF file with error handling and metrics.

    Returns:
        Status dictionary with processing results.
    """
    status = {
        "file": path,
        "filename": os.path.basename(path),
        "text_extracted": False,
        "num_chunks": 0,
        "embedded": False,
        "error": None
    }

    try:
        store = ingest_pdf(path)

        if store is None:
            raise RuntimeError("PDF ingestion failed")
        status["store"] = store
        status["text_extracted"] = True
        status["num_chunks"] = len(store.metadata)
        status["embedded"] = True

        logger.info(f"Successfully processed {path}: ")

    except Exception as e:
        logger.error(f"Error processing {path}: {str(e)}")
        status["error"] = str(e)
    return status
