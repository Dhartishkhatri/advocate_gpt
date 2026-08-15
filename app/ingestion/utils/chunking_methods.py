import os

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
    """
    Split extracted PDF text into overlapping chunks for embedding.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks

def chunk_by_words(text, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - chunk_overlap

    return chunks




def chunk_by_tokens(text, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP, model="gpt-4o-mini"):
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))

        if end == len(tokens):
            break

        start = end - chunk_overlap

    return chunks



def chunk_by_sentences(text, max_chars=1000):
    import nltk
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks

def chunk_by_paragraphs(text, max_chars=1500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) <= max_chars:
            current += "\n\n" + paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks

def chunk_by_custom_splitter(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into chunks using a custom splitter (e.g., langchain's RecursiveCharacterTextSplitter).

    parameters:
    -----------
    text : str
        The full text of the document to be chunked.
    chunk_size : int
        The maximum number of characters in each chunk.
    chunk_overlap : int
        The number of characters to overlap between consecutive chunks.
    
    returns:
    --------
    chunks : list
        A list of text chunks.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks

def chunk_by_adding_contextual_document_header(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, header=""):
    """
    Split text into chunks and add a contextual header to each chunk.
    """
    base_chunks = chunk_text(text, chunk_size, chunk_overlap)
    header = header.strip()

    chunks_with_headers = []
    for i, chunk in enumerate(base_chunks):
        chunks_with_headers.append(f"Document Title: {header}\n\n{chunk}")

    return chunks_with_headers


def generate_chunk_title(chunk_text: str, chunk_title_guidance: str = "") -> str:
    """
    Extract a chunk title using OpenAI.
    """
    from openai import OpenAI
    import tiktoken

    max_content_tokens = 4000
    model_name = "gpt-4o-mini"
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    tokens = encoder.encode(chunk_text, disallowed_special=())
    truncated_text = encoder.decode(tokens[:max_content_tokens])
    truncation_message = ""

    if len(tokens) >= max_content_tokens:
        truncation_message = (
            "Also note that the chunk text provided below is just the first "
            "~3000 words of the chunk. That should be plenty for this task. "
            "Your response should still pertain to the entire chunk, not just "
            "the text provided below."
        )

    prompt = f"""
INSTRUCTIONS
What is the title of the following chunk?

Your response MUST be the title of the chunk, and nothing else. DO NOT respond with anything else.

{chunk_title_guidance}

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

def chunk_by_adding_contextual_chunk_header(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into chunks and add a contextual header to each chunk.

    Parameters:
    -----------
    text : str
        The full text of the document to be chunked.
    chunk_size : int
        The maximum number of characters in each chunk.
    chunk_overlap : int
        The number of characters to overlap between consecutive chunks.
    
    Returns:
    --------
    chunks_with_headers : list
        A list of text chunks with contextual headers.
    """
    base_chunks = chunk_text(text, chunk_size, chunk_overlap)
    

    chunks_with_headers = []
    for i, chunk in enumerate(base_chunks):
        header = generate_chunk_title(chunk)
        header = header.strip()
        chunks_with_headers.append(f"Chunk Title: {header}\n\n{chunk}")

    return chunks_with_headers

def chunk_by_semantic(text):
    """
    Split text into semantically meaningful chunks using OpenAI embeddings.
    Three breakpoint types are available:
    'percentile': Splits at differences greater than the X percentile.
    'standard_deviation': Splits at differences greater than X standard deviations.
    'interquartile': Uses the interquartile distance to determine split points.

    Parameters:
    -----------
    text : str
        The full text of the document to be chunked.

    Returns:
    --------
    chunks : list
        A list of semantically meaningful text chunks.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai.embeddings import OpenAIEmbeddings

    text_splitter = SemanticChunker(OpenAIEmbeddings(model = 'text-embedding-3-small'), breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90) # chose which embeddings and breakpoint type and threshold to use
    chunks = text_splitter.split_text(text)
    return chunks