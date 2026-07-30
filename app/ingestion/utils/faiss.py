import os
import json
import faiss
import numpy as np

from app.config import VECTOR_STORE_PATH, METADATA_PATH, EMBEDDING_DIM


try:
    _global_index = faiss.read_index(VECTOR_STORE_PATH)
    _global_metadata = json.load(open(METADATA_PATH))
except:
    _global_index = None
    _global_metadata = []
    
class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, docs):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(docs)

    def search(self, query_embedding, k=5):
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, k)
        return [self.metadata[i] for i in I[0]]

    def save(self, index_path=VECTOR_STORE_PATH, metadata_path=METADATA_PATH):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    @staticmethod
    def load(index_path=VECTOR_STORE_PATH, metadata_path=METADATA_PATH):
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            index = faiss.read_index(index_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            store = FAISSStore(index.d)
            store.index = index
            store.metadata = metadata
            return store
        return None