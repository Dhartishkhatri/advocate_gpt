import json
from pathlib import Path
from collections import defaultdict
import os
import re
from collections import Counter
from typing import List, Dict, Any
import faiss
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
from app.ingestion.utils.knowledge_graph_methods import build_knowledge_graph
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
    """
    Use the same tokenizer when indexing and searching.
    Parameters:
    -----------
    text : str
        The text to tokenize.

    Returns:
    --------
    tokens : list
        The list of tokens.
    """

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def create_bm25_index(chunks, index_path=None):
    """
    Create a term-to-chunks inverted BM25 index.
    parameters:
    -----------
    chunks : list
        List of chunk dictionaries with 'id' and 'text'.
    index_path : str, optional
        Path to save the BM25 index as a JSON file. If None, the index is not saved.

    returns:
    --------
    index : dict
        The BM25 index containing postings, chunk lengths, and frequencies.
    """

    postings = defaultdict(list)
    chunk_lengths = {}
    total_chunk_length = 0

    for chunk in chunks:
        chunk_id = str(chunk.get("id", 'unknown'))
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
    chunk_frequencies = {term: len(term_postings) for term, term_postings in postings.items()}

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


def update_bm25_index(chunks, index_path, commit=True):
    """
    Update an existing BM25 index with new chunks.
    parameters:
    -----------
    new_chunks : list
        List of new chunk dictionaries with 'id' and 'text'.
    index_path : str
        Path to the existing BM25 index JSON file.
    
    returns:
    --------
    index : dict
        The updated BM25 index.
    added_count : int
        The number of new chunks added to the index.
    """

    with index_path.open("r", encoding="utf-8") as index_file:
        index = json.load(index_file)

    existing_chunk_ids = set(index["chunk_lengths"])
    added_count = 0

    for chunk in chunks:
        chunk_id = str(chunk.get("id", 'unknown'))

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
            index["postings"].setdefault(term, []).append([chunk_id, term_frequency])
            # A Counter contains each term once, so this increments once per
            # new chunk containing the term.
            index["chunk_frequencies"][term] = index["chunk_frequencies"].get(term, 0) + 1

        existing_chunk_ids.add(chunk_id)
        added_count += 1

    index["average_chunk_length"] = (index["total_chunk_length"] / index["chunk_count"] if index["chunk_count"] else 0.0)

    # Pass commit=False when the caller publishes this alongside the vector store.
    if commit:
        save_bm25_index(index, index_path)

    return index, added_count


def save_bm25_index(index, index_path):
    """Write BM25 atomically to avoid a partially written JSON file.
    parameters:
    -----------
    index : dict
        The BM25 index to save.
    
    index_path : str
        Path to save the BM25 index as a JSON file.

    returns:
    --------
    None    
    """

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as index_file:
        json.dump(index, index_file, ensure_ascii=False)
        index_file.flush()
        os.fsync(index_file.fileno())

    os.replace(temporary_path, index_path)


def save_all_indexes(vector_store, bm25_index, knowledge_graph=None):
    """Save the FAISS store, the BM25 index and the knowledge graph together, or not at all.

    Every file is written as ".tmp" first, so nothing is published until all of
    them are complete. They are then swapped in the order that keeps each
    intermediate state usable: extra metadata rows are simply never looked up,
    extra vectors would break a metadata lookup, and the two chunk-id based
    indexes go last so they can never point at chunks the vector store does not
    have yet.
    """

    Path(config.VECTOR_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)

    json_targets = [
        (vector_store.metadata, config.METADATA_PATH),
        (bm25_index, config.BM25_INDEX_PATH),
    ]
    replace_order = [config.METADATA_PATH, config.VECTOR_STORE_PATH, config.BM25_INDEX_PATH]

    if knowledge_graph is not None:
        json_targets.append((knowledge_graph.to_dict(), config.KNOWLEDGE_GRAPH_PATH))
        replace_order.append(config.KNOWLEDGE_GRAPH_PATH)

    faiss.write_index(vector_store.index, f"{config.VECTOR_STORE_PATH}.tmp")
    for data, path in json_targets:
        with open(f"{path}.tmp", "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, ensure_ascii=False)

    for path in replace_order:
        os.replace(f"{path}.tmp", path)


def ingest_pdf(file_path):
    """
    Full PDF ingestion pipeline:
    1. Extract PDF text
    2. Chunk text
    3. Embed chunks
    4. Store embeddings in FAISS
    5. Extend the knowledge graph with the new chunks
    6. Save FAISS index, BM25 index and knowledge graph

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

        # Build both indexes in memory only, so a later failure leaves disk untouched.
        if bm25_index_path.is_file():
            bm25_index, bm25_added = update_bm25_index(chunks=all_docs, index_path=bm25_index_path, commit=False)
        else:
            bm25_index = create_bm25_index(all_docs)
            bm25_added = bm25_index["chunk_count"]
        logger.info(f"BM25 index covers {bm25_added} new chunks")

        vector_store = FAISSStore.load()
        embeddings = chunk_to_embed(all_docs)

        if len(embeddings) == 0:
            raise ValueError("No embeddings were generated")
        dim = len(embeddings[0])
        if vector_store is None:
            vector_store = FAISSStore(dim)
        vector_store.add(embeddings=embeddings, docs=all_docs)

        knowledge_graph = None
        if config.BUILD_KNOWLEDGE_GRAPH:
            try:
                # commit=False so the graph is published with the other indexes
                # below rather than written on its own.
                knowledge_graph = build_knowledge_graph(
                    all_docs, embeddings=embeddings, store=vector_store, commit=False
                )
            except Exception as exc:
                # The graph is an auxiliary index; losing it must not cost us the
                # chunks that were successfully embedded.
                logger.warning(f"Skipping knowledge graph for {file_path}: {exc}")

        save_all_indexes(vector_store, bm25_index, knowledge_graph)

        return vector_store,len(all_docs)

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
        store, num_chunks = ingest_pdf(file_path)

        if store is None:
            raise RuntimeError("PDF ingestion failed")
        status["store"] = store
        status["text_extracted"] = True
        status["num_chunks"] = num_chunks
        status["embedded"] = True

        logger.info(f"Successfully processed {file_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        status["error"] = str(e)
    return status
