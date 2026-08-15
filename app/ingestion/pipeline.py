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



def create_bm25_index(chunks, index_path=None):
    """
    Create a BM25 index from the provided documents.

    Parameters:
    -----------
    chunks : list
        List of chunk dictionaries with 'text' field.
    index_path : str (optional)
        Path to save the BM25 index as a JSON file. If None, the index is not saved.

    Returns:
    --------
    index : dict
        BM25 index containing term frequencies, chunk frequencies, and other metadata.
    """

    tokenized_chunks = [re.findall(r"\w+", chunk.get("text", "").lower()) for chunk in chunks]

    term_frequencies = [dict(Counter(tokens)) for tokens in tokenized_chunks]

    chunk_frequencies = Counter(term for tokens in tokenized_chunks for term in set(tokens))
    
    chunk_length = [len(tokens) for tokens in tokenized_chunks]

    index = {
        "chunk_count": len(chunks),
        "average_chunk_length": (sum(chunk_length) / len(chunks) if chunks else 0),
        "chunk_lengths": chunk_length,
        "term_frequencies": term_frequencies,
        "chunk_frequencies": dict(chunk_frequencies),
    }

    return index

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


        os.makedirs("vector_store", exist_ok=True)
        
        # Create and save BM25 index
        bm25_index = create_bm25_index(all_docs)
        with open(config.BM25_INDEX_PATH, "w", encoding="utf-8") as index_file:
            json.dump(bm25_index, index_file)

        # Create and save embeddings
        embeddings = chunk_to_embed(all_docs)
        if len(embeddings) == 0:
            raise ValueError("No embeddings were generated")
        dim = len(embeddings[0])
        store = FAISSStore(dim)
        store.add(embeddings=embeddings,docs=all_docs)
        store.save()

        return store

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
