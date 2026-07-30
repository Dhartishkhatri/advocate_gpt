"""
Flask application for Advocate GPT with ChatGPT-like interface
Production-ready version with comprehensive error handling and monitoring
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import time
from dotenv import load_dotenv
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer
from app.logger import Logger
import traceback
from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TIMEOUT

# Load environment variables
load_dotenv()

# Initialize logging
logger = Logger.get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Store conversation history per session (in production, use database)
conversations = {}


@app.route('/')
def index():
    """Serve the main chat interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return None


@app.route('/api/query', methods=['POST'])
def query():
    """
    Handle user queries through the RAG system
    
    Request JSON:
    {
        "query": "user question",
        "session_id": "unique session identifier" (optional)
    }
    
    Response JSON:
    {
        "answer": "generated answer",
        "sources": [list of source documents],
        "session_id": "session identifier",
        "metadata": {
            "retrieval_time_ms": float,
            "generation_time_ms": float,
            "hallucination_checked": bool
        },
        "error": "error message if any"
    }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            logger.warning("Empty query received")
            return jsonify({'error': 'Query cannot be empty'}), 400
        

        try:
            docs = retrieve(user_query)
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return jsonify({'error': 'Error retrieving documents. Please try again.'}), 500
        
        if not docs:
            logger.warning(f"No documents retrieved for query: {user_query}")
            return jsonify({
                'answer': 'No relevant documents found. Please try rephrasing your question.',
                'sources': []
                })
        

        try:
            result = generate_answer(user_query, docs)
            answer = result.get('answer', '')
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': 'Error generating answer. Please try again.'}), 500

        
        # Format sources for response with improved information
        sources = []
        for doc in docs:
            source_doc = {
                'text': doc.get('text', '')[:300] + '...' if len(doc.get('text', '')) > 300 else doc.get('text', ''),
                'source': doc.get('source', 'Unknown'),
                'page': doc.get('page', 'N/A'),
                'section': doc.get('section', 'N/A'),
                'relevance_score': doc.get('relevance_score', 'N/A')
            }
            sources.append(source_doc)
        
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'metadata': {
                'documents_retrieved': len(docs)
            }
        })
    
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        return None

if __name__ == '__main__':
    # Ensure templates folder exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info(f"Starting Advocate GPT on {LLM_PROVIDER} with Ollama model {OLLAMA_MODEL}")

    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
