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
    "extract_text_from_pdf_using_pdfminer": extract_text_from_pdf_using_pdfminer
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
    "chunk_by_semantic": chunk_by_semantic
}

from app.logger import Logger
logger = Logger.get_logger(__name__)

def normalize_chunks(chunks, file_path):
    """
    Convert chunks into standard document format for FAISS metadata.

    Parameters:
    -----------
    chunks : list
        List of text chunks extracted from the PDF.
    file_path : str
        Path to the original PDF file.

    Returns:
    --------
    normalized_docs : list
        List of normalized document dictionaries with 'id', 'text', and 'metadata'.
    """
    normalized_docs = []

    for i, chunk in enumerate(chunks):
        text = str(chunk).strip()
        if not text:
            continue

        normalized_docs.append({
            "id": f"{os.path.basename(file_path)}_{i}",
            "text": text,
            "metadata": {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "chunk_id": i,
            }
        })

    return normalized_docs


def chunk_to_embed(docs):
    """
    Create embeddings for all chunks.

    Parameters:
    -----------
    docs : list
        List of document dictionaries with 'text' field.

    Returns:
    --------
    embeddings : list
        List of embeddings corresponding to the document chunks.
    """

    texts = [doc["text"] for doc in docs]

    if not texts:
        return []

    embeddings = embed_batch(texts)

    return embeddings


def generate_document_title(document_text, document_title_guidance="") -> str:
    """
    Extract a document title using OpenAI.

    Parameters:
    -----------
    document_text : str
        The text of the document from which to extract the title.
    document_title_guidance : str
        Optional guidance for the title extraction.

    Returns:
    --------
    title : str
        The extracted title of the document.
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
            "~4000 words of the document. That should be plenty for this task. "
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

def tokenize_bm25(text):
    """Use the same tokenizer when indexing and searching."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def create_bm25_index(chunks, index_path=None):
    """Create a term-to-chunks inverted BM25 index.

    The important fields have this shape:

        chunk_lengths[chunk_id] = number of tokens in the chunk
        postings[term] = [[chunk_id, term_frequency], ...]
        chunk_frequencies[term] = number of chunks containing the term

    Lists are used instead of tuples in postings because JSON serializes tuples
    as lists anyway.
    """
    postings = defaultdict(list)
    chunk_lengths = {}
    total_chunk_length = 0

    for fallback_position, chunk in enumerate(chunks):
        chunk_id = str(chunk.get("id", fallback_position))
        tokens = tokenize_bm25(chunk.get("text", ""))

        if not tokens:
            continue

        term_frequencies = Counter(tokens)
        chunk_length = len(tokens)

        chunk_lengths[chunk_id] = chunk_length
        total_chunk_length += chunk_length

        # Each term is added once for this chunk, together with its TF.
        for term, term_frequency in term_frequencies.items():
            postings[term].append([chunk_id, term_frequency])

    chunk_count = len(chunk_lengths)

    # One posting exists for every distinct chunk containing the term.
    chunk_frequencies = {
        term: len(term_postings)
        for term, term_postings in postings.items()
    }

    index = {
        "version": 1,
        "chunk_count": chunk_count,
        "total_chunk_length": total_chunk_length,
        "average_chunk_length": (
            total_chunk_length / chunk_count if chunk_count else 0.0
        ),
        "chunk_lengths": chunk_lengths,
        "postings": dict(postings),
        "chunk_frequencies": chunk_frequencies,
    }

    if index_path is not None:
        save_bm25_index(index, index_path)

    return index


def update_bm25_index(new_chunks, index_path):
    """Append previously unseen chunks to an existing BM25 JSON index.

    This operation is idempotent when chunk IDs are stable: a chunk already in
    ``chunk_lengths`` is skipped. Use one writer process, and call this with a
    reasonably large batch instead of rewriting the JSON file for every PDF.

    Returns:
        tuple[dict, int]: Updated index and number of newly added chunks.
    """
    with index_path.open("r", encoding="utf-8") as index_file:
        index = json.load(index_file)

    existing_chunk_ids = set(index["chunk_lengths"])
    added_count = 0

    for fallback_position, chunk in enumerate(new_chunks):
        chunk_id = str(chunk.get("id", fallback_position))

        if chunk_id in existing_chunk_ids:
            continue

        tokens = tokenize_bm25(chunk.get("text", ""))
        if not tokens:
            continue

        term_frequencies = Counter(tokens)
        chunk_length = len(tokens)

        index["chunk_lengths"][chunk_id] = chunk_length
        index["chunk_count"] += 1
        index["total_chunk_length"] += chunk_length

        for term, term_frequency in term_frequencies.items():
            index["postings"].setdefault(term, []).append(
                [chunk_id, term_frequency]
            )
            # A Counter contains each term once, so this increments once per
            # new chunk containing the term.
            index["chunk_frequencies"][term] = (
                index["chunk_frequencies"].get(term, 0) + 1
            )

        existing_chunk_ids.add(chunk_id)
        added_count += 1

    index["average_chunk_length"] = (
        index["total_chunk_length"] / index["chunk_count"]
        if index["chunk_count"]
        else 0.0
    )

    save_bm25_index(index, index_path)
    return index, added_count



def ingest_pdf(file_path):
    """
    Full PDF ingestion pipeline:
    1. Extract PDF text
    2. Chunk text
    3. Embed chunks
    4. Store embeddings in FAISS
    5. Save FAISS index

    Parameters:
    -----------
    file_path : str
        Path to the PDF file to be ingested.
    
    Returns:
    --------
    store : FAISSStore or None
        Returns FAISSStore object if successful, None if an error occurred.
    """

    logger.info(f"Starting ingestion pipeline for: {file_path}")
    try:
        pdf_text_extraction_method = pdf_text_extraction_methods.get(config.PDF_TEXT_EXTRACTION_METHOD)
        text = pdf_text_extraction_method(file_path)

        chunking_method = pdf_chunking_methods.get(config.PDF_CHUNKING_METHOD)
        if config.PDF_CHUNKING_METHOD == "chunk_by_adding_contextual_chunk_header":
            header = generate_document_title(text)
            chunks = chunking_method(text,header)        
        else:
            chunks = chunking_method(text)

        if not chunks:
            raise ValueError("No text chunks were extracted from the PDF")

        all_docs = normalize_chunks(chunks,file_path)


        bm25_index_path = Path(config.BM25_INDEX_PATH)

        if bm25_index_path.is_file():
            _, bm25_added = update_bm25_index(new_chunks=all_docs,index_path=bm25_index_path)
            logger.info("Updated BM25 index with %d chunks", bm25_added)
        else:
            bm25_index = create_bm25_index(all_docs)
            save_bm25_index(bm25_index, bm25_index_path)
            bm25_added = bm25_index["chunk_count"]
            logger.info("Created BM25 index with %d chunks", bm25_added)


        vector_store = FAISSStore.load()
        embeddings = chunk_to_embed(all_docs)

        if len(embeddings) == 0:
            raise ValueError("No embeddings were generated")
        dim = len(embeddings[0])
        vector_store = FAISSStore(dim)
        vector_store.add(embeddings=embeddings, docs=all_docs)
        vector_store.save()

        return vector_store

    except Exception as e:
        logger.error(f"Error during {file_path} ingestion: {str(e)}")
        return None


def process_pdf(file_path):
    """
    Process a single PDF file with error handling and metrics.

    Parameters:
    -----------
    file_path : str
        Path to the PDF file to be processed.

    Returns:
    --------
    status : dict
        Status dictionary with processing results.
    """
    status = {
        "file": file_path,
        "filename": os.path.basename(file_path),
        "text_extracted": False,
        "num_chunks": 0,
        "embedded": False,
        "error": None
    }

    try:
        store = ingest_pdf(file_path)

        if store is None:
            raise RuntimeError("PDF ingestion failed")
        status["store"] = store
        status["text_extracted"] = True
        status["num_chunks"] = len(store.metadata)
        status["embedded"] = True

        logger.info(f"Successfully processed {file_path}: ")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        status["error"] = str(e)
    return status
