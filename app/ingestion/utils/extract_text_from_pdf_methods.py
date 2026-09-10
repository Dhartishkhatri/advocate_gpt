def extract_text_from_pdf_using_pypdf(file_path):
    """
    Extract text from PDF using pypdf. Use pypdf when you want to change the PDF file.
    Examples: merge PDFs, split pages, rotate pages, add password, crop pages, read metadata. 
    The official project describes it as a pure-Python library for splitting, merging, cropping, transforming pages,
    adding passwords, and retrieving text/metadata.

    parameters:
    -----------
    file_path (str): The path to the PDF file.

    Returns:
    -----------
    str: The extracted text from the PDF file.
    """

    from pypdf import PdfReader
    elements = PdfReader(file_path)
    text = ""
    for page in elements.pages:
        text += page.extract_text() or ""

    return text.strip()


def extract_text_from_pdf_using_unstructured(file_path):
    """
    Extract text from PDF using unstructured.
    This method extracts text from the PDF while preserving the structure of the document, including titles, narrative text, 
    and tables. 
    Use this method when you need structured chunks, metadata, and support for multiple file types.
    parameters:
    -----------
    file_path (str): The path to the PDF file.

    Returns:
    -----------
    str: The extracted text from the PDF file.
    """
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Title, NarrativeText, Table

    elements = partition_pdf(filename=file_path)

    extracted_parts = []

    for element in elements:
        if isinstance(element, (Title, NarrativeText, Table)):
            text = str(element).strip()
            if text:
                extracted_parts.append(text)
    full_text = "\n\n".join(extracted_parts)

    return full_text

def extract_text_from_pdf_using_pdfplumber(file_path):
    """
    Extract text from PDF using pdfplumber.
    Use pdfplumber when you want to extract data from PDFs, especially tables.
    It exposes detailed PDF layout information such as characters, rectangles, lines, and tables, and works best on 
    machine-generated PDFs rather than scanned PDFs.
    """
    import pdfplumber

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables = page.extract_tables()

    return text.strip()

def extract_text_from_pdf_using_pymupdf(file_path):
    """
    Extract text from PDF using PyMuPDF (fitz).
    Use PyMuPDF when you want speed, rendering, images, annotations, or general extraction.
    PyMuPDF describes itself as a high-performance library for extraction, analysis, conversion, and manipulation of PDF and
    other document formats.
    It also supports image extraction and OCR workflows with Tesseract installed separately.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text.strip()

def extract_text_from_pdf_using_pdfminer(file_path):
    """
    Extract text from PDF using pdfminer.
    Use pdfminer when text exists in the PDF, but normal extraction gives badly ordered or badly spaced text, 
    and you need to tune the extraction behavior.
    """
    from pdfminer.high_level import extract_text

    text = extract_text(file_path)

    return text.strip()

def extract_text_from_pdf_using_tesseract(file_path):
    """
    Extract text from scanned PDF using Tesseract OCR. 
    Use Tesseract when the PDF/image contains text that is not selectable.
    That means the document is basically an image, not real embedded PDF text.
    Extract text from scanned PDF using Tesseract OCR. 
    """
    from pdf2image import convert_from_path
    import pytesseract

    pages = convert_from_path(file_path)

    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)

    return text.strip()
