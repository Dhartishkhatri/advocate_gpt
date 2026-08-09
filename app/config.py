"""
Application configuration.

Change PDF_TEXT_EXTRACTION_METHOD or PDF_CHUNKING_METHOD to switch the
ingestion pipeline behavior.
"""

DATA_FOLDER = "pdfs/"

PDF_TEXT_EXTRACTION_METHOD = "extract_text_from_pdf_using_unstructured"
pdf_text_extraction_methods = (
    "extract_text_from_pdf_using_pypdf",
    "extract_text_from_pdf_using_unstructured",
    "extract_text_from_pdf_using_pdfplumber",
    "extract_text_from_pdf_using_pymupdf",
    "extract_text_from_pdf_using_tesseract",
    "extract_text_from_pdf_using_pdfminer",
)

PDF_CHUNKING_METHOD = "chunk_by_custom_splitter"
pdf_chunking_methods = (
    "chunk_text",
    "chunk_by_words",
    "chunk_by_tokens",
    "chunk_by_sentences",
    "chunk_by_paragraphs",
    "chunk_by_custom_splitter",
    "chunk_by_adding_contextual_chunk_header",
    "chunk_by_adding_contextual_document_header",
    "chunk_by_semantic",
)

RETRIEVAL_METHOD = "hierarchical_retrieval"
retrieval_methods = (
    "bm25_retrieval",
    "fusion_retrieval",
    "get_best_segments",
    "hierarchical_retrieval",
)

RERANKING_METHOD = "greedy_dartboard_search"
reranking_methods = (
    "greedy_dartboard_search",
    "weighted_reciprocal_rank_fusion",
)


FUSION_ALPHA = 0.5
HIERARCHICAL_SUMMARY_K = 3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
VECTOR_STORE_PATH = "vector_store/faiss.index"
METADATA_PATH = "metadata.json"
SUMMARY_VECTOR_STORE_PATH = "vector_store/summary_faiss.index"
SUMMARY_METADATA_PATH = "vector_store/summary_metadata.json"
BM25_INDEX_PATH = "vector_store/bm25_index.json"
RSE_MAX_SEGMENT_LENGTH = 3
RSE_MINIMUM_VALUE = 0.01

SUMMARY_MODEL = "gpt-4o"
SUMMARY_CONCURRENCY = 5
SUMMARY_MAX_INPUT_CHARS = 12000

LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.7
MAX_TOKENS = 1000
TIMEOUT = 300