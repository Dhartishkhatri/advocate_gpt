import os
from app.config import *
from app.ingestion.pipeline import process_pdf
from app.logger import Logger
logger = Logger.get_logger(__name__)


PROCESSED = set()

def scan_folder():
    """
    Scan PDF folder and ingest all documents.
    Includes error handling and progress tracking.
    parameters:
    -----------
    None

    returns:
    --------
    results : list
        List of ingestion results (dictionary) for each PDF file.
    """
    logger.info(f"Scanning folder: {DATA_FOLDER}")
    logger.info(f"Using text extraction method: {PDF_TEXT_EXTRACTION_METHOD}")
    logger.info(f"Using chunking method: {PDF_CHUNKING_METHOD}")
    logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Vector store path: {VECTOR_STORE_PATH}")

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    results = []
    successful = 0
    failed = 0
    
    for file in pdf_files:
        if file not in PROCESSED:
            file_path = os.path.join(DATA_FOLDER, file)
            
            try:
                logger.info(f"Ingesting: {file}")
                status = process_pdf(file_path)
                results.append(status)
                
                if status.get("error"):
                    logger.error(f"Failed to ingest {file}: {status['error']}")
                    failed += 1
                else:
                    logger.info(f"Successfully ingested {file}: {status['num_chunks']} chunks")
                    successful += 1
                
                PROCESSED.add(file)
            
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")
                failed += 1
                PROCESSED.add(file)
    
    logger.info(f"Ingestion complete: {successful} successful, {failed} failed")
    return results
