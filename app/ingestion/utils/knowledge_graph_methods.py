"""
Knowledge graph construction over ingested chunks.

A node is one chunk, keyed by the chunk id produced by
``app.ingestion.pipeline.normalize_chunks``, so a graph can be extended across
ingestion runs without the node ids shifting. An edge joins two chunks whose
embeddings are similar, and its weight blends that similarity with the concepts
the two chunks share.

Concepts come from two sources: named entities extracted with spaCy, and general
concepts extracted with the configured LLM. Both are optional at runtime — if
spaCy is not installed, or the LLM is unreachable, extraction falls back to a
frequency based keyword pass so the graph is still built.
"""

import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from app import config
from app.logger import Logger

logger = Logger.get_logger(__name__)


# Words that carry no meaning as a shared concept between two legal chunks.
STOP_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "may", "not", "of", "on", "or", "our", "shall", "she", "should",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "was", "were", "which", "who", "will", "with",
    "would", "you", "your",
}


def _load_networkx():
    """
    Import networkx, reporting a clear error when it is not installed.

    Parameters:
    -----------
    None

    Returns:
    --------
    networkx : module
        The imported networkx module.
    """

    try:
        import networkx
    except ImportError as exc:
        raise ImportError(
            "networkx is required to build the knowledge graph. "
            "Install it with: pip install networkx"
        ) from exc

    return networkx


def _load_spacy_model():
    """
    Load the spaCy model named in config, downloading it if necessary.

    Parameters:
    -----------
    None

    Returns:
    --------
    nlp : spacy.Language or None
        The loaded spaCy pipeline, or None when spaCy is unavailable. None means
        concept extraction falls back to keywords instead of failing.
    """

    try:
        import spacy
    except ImportError as exc:
        # Not only a missing package: a native module spaCy loads can also fail,
        # so the reason is reported rather than assumed.
        logger.warning(f"spaCy unavailable, named entity extraction disabled: {exc}")
        return None

    model_name = getattr(config, "KNOWLEDGE_GRAPH_SPACY_MODEL", "en_core_web_sm")

    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model {model_name}")
        try:
            from spacy.cli import download

            download(model_name)
            return spacy.load(model_name)
        except Exception as exc:
            logger.warning(f"Could not load spaCy model {model_name}: {exc}")
            return None


def normalize_concept(concept):
    """
    Normalize a concept so the same idea written two ways matches as shared.

    Parameters:
    -----------
    concept : str
        The raw concept or entity string.

    Returns:
    --------
    normalized : str
        Lowercased, whitespace collapsed, punctuation trimmed concept. Empty
        when nothing meaningful is left.
    """

    normalized = re.sub(r"\s+", " ", str(concept)).strip().lower()
    normalized = normalized.strip(".,;:!?\"'()[]{}-")

    if len(normalized) < 3 or normalized in STOP_WORDS:
        return ""

    return normalized


def _unit_vectors(embeddings):
    """
    Return the embeddings as L2 normalized float32 rows.

    Parameters:
    -----------
    embeddings : array-like
        A 2D array of embeddings, one row per chunk.

    Returns:
    --------
    vectors : numpy.ndarray
        The normalized embeddings.
    """

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # embed_batch already L2 normalizes, but a caller may pass raw vectors.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


def cosine_similarity_matrix(embeddings):
    """
    Compute the pairwise cosine similarity matrix for a set of embeddings.

    Parameters:
    -----------
    embeddings : array-like
        A 2D array of embeddings, one row per chunk.

    Returns:
    --------
    similarities : numpy.ndarray
        A square matrix of cosine similarities.
    """

    vectors = _unit_vectors(embeddings)

    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    return vectors @ vectors.T


def _cross_similarities(embeddings, other_embeddings):
    """
    Compute cosine similarities between two separate sets of embeddings.

    Parameters:
    -----------
    embeddings : array-like
        One embedding row per new chunk.
    other_embeddings : array-like
        One embedding row per existing chunk.

    Returns:
    --------
    similarities : numpy.ndarray
        A matrix with one row per new chunk and one column per existing chunk.
    """

    vectors = _unit_vectors(embeddings)
    other_vectors = _unit_vectors(other_embeddings)

    if vectors.size == 0 or other_vectors.size == 0:
        return np.zeros((vectors.shape[0], other_vectors.shape[0]), dtype=np.float32)

    return vectors @ other_vectors.T


def load_store_embeddings(chunk_ids, store=None):
    """
    Read the vectors of already ingested chunks back out of the FAISS store.

    Reusing the stored vectors is what lets a new PDF link to chunks from
    earlier ingestion runs without embedding the whole corpus again.

    Parameters:
    -----------
    chunk_ids : iterable
        The chunk ids to look up.
    store : FAISSStore, optional
        An open store. Loaded from disk when omitted.

    Returns:
    --------
    found_ids : list
        The ids that were present in the store.
    vectors : numpy.ndarray
        The vectors for those ids, in the same order.
    """

    wanted = {str(chunk_id) for chunk_id in chunk_ids}

    if not wanted:
        return [], np.zeros((0, 0), dtype=np.float32)

    if store is None:
        from app.ingestion.utils.faiss import FAISSStore

        store = FAISSStore.load()

    if store is None:
        return [], np.zeros((0, 0), dtype=np.float32)

    found_ids = []
    vectors = []

    for row, document in enumerate(store.metadata):
        chunk_id = str(document.get("id", ""))

        if chunk_id in wanted and row < store.index.ntotal:
            found_ids.append(chunk_id)
            vectors.append(store.index.reconstruct(row))

    if not found_ids:
        return [], np.zeros((0, 0), dtype=np.float32)

    return found_ids, np.asarray(vectors, dtype=np.float32)


def extract_keywords(text, max_concepts):
    """
    Extract concepts by term frequency, used when spaCy and the LLM are absent.

    Parameters:
    -----------
    text : str
        The chunk text.
    max_concepts : int
        The maximum number of keywords to return.

    Returns:
    --------
    keywords : list
        The most frequent meaningful terms in the text.
    """

    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
        if token not in STOP_WORDS
    ]

    return [term for term, _ in Counter(tokens).most_common(max_concepts)]


def extract_concepts_with_ollama(text, max_concepts):
    """
    Ask the configured Ollama model for the key concepts in a chunk.

    Parameters:
    -----------
    text : str
        The chunk text.
    max_concepts : int
        The maximum number of concepts to request.

    Returns:
    --------
    concepts : list
        The extracted concepts, or an empty list when the model is unreachable
        or returns something that is not the expected JSON shape.
    """

    import requests

    prompt = f"""You are a legal analyst extracting index terms.

List up to {max_concepts} key concepts from the text below.
Exclude named entities such as people, organizations and places.
Respond with JSON only, in the form {{"concepts": ["concept one", "concept two"]}}.

TEXT
{text[: config.KNOWLEDGE_GRAPH_MAX_INPUT_CHARS]}"""

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.0,
            },
            timeout=config.TIMEOUT,
        )
    except Exception as exc:
        logger.warning(f"Concept extraction request failed: {exc}")
        return []

    if response.status_code != 200:
        logger.warning(
            f"Concept extraction error: {response.status_code} - {response.text}"
        )
        return []

    try:
        payload = json.loads(response.json().get("response", "{}"))
    except (ValueError, AttributeError) as exc:
        logger.warning(f"Could not parse concept extraction response: {exc}")
        return []

    concepts = payload.get("concepts", payload.get("concepts_list", []))

    if not isinstance(concepts, list):
        return []

    return [str(concept) for concept in concepts][:max_concepts]


class KnowledgeGraph:
    """
    A graph of chunks linked by embedding similarity and shared concepts.

    Attributes:
    -----------
    graph : networkx.Graph
        The graph itself; nodes carry 'text' and 'concepts', edges carry
        'weight', 'similarity' and 'shared_concepts'.
    concept_cache : dict
        Chunk text to extracted concepts, so repeated text is only sent to the
        LLM once.
    nlp : spacy.Language or None
        The spaCy pipeline used for named entity extraction, loaded lazily.
    edges_threshold : float
        The minimum cosine similarity for two chunks to be joined by an edge.
    """

    def __init__(self, edges_threshold=None):
        """
        Initialize an empty knowledge graph.

        Parameters:
        -----------
        edges_threshold : float, optional
            Overrides config.KNOWLEDGE_GRAPH_EDGE_THRESHOLD.

        Returns:
        --------
        None
        """

        networkx = _load_networkx()

        self.graph = networkx.Graph()
        self.concept_cache = {}
        self._nlp = None
        self._nlp_loaded = False
        # Concept extraction is threaded, and the first access may download the
        # spaCy model; without this every worker would start its own download.
        self._nlp_lock = threading.Lock()
        self.edges_threshold = (
            config.KNOWLEDGE_GRAPH_EDGE_THRESHOLD
            if edges_threshold is None
            else edges_threshold
        )

    @property
    def nlp(self):
        """
        The spaCy pipeline, loaded on first use.

        Parameters:
        -----------
        None

        Returns:
        --------
        nlp : spacy.Language or None
            The pipeline, or None when spaCy is unavailable.
        """

        if not self._nlp_loaded:
            with self._nlp_lock:
                if not self._nlp_loaded:
                    self._nlp = _load_spacy_model()
                    self._nlp_loaded = True

        return self._nlp



    def _add_nodes(self, docs):
        """
        Add one node per chunk, keyed by chunk id.

        Parameters:
        -----------
        docs : list
            Normalized chunk dictionaries with 'id' and 'text'.

        Returns:
        --------
        None
        """

        for doc in docs:
            node_id = str(doc["id"])
            # Re-ingesting a PDF must not wipe concepts already extracted for it.
            existing_concepts = (
                self.graph.nodes[node_id].get("concepts", [])
                if node_id in self.graph
                else []
            )
            self.graph.add_node(
                node_id, text=doc.get("text", ""), concepts=existing_concepts
            )

    def _compute_similarities(self, embeddings):
        """
        Compute the cosine similarity matrix for the embeddings.

        Parameters:
        -----------
        embeddings : array-like
            One embedding row per chunk.

        Returns:
        --------
        similarities : numpy.ndarray
            A square matrix of cosine similarities.
        """

        return cosine_similarity_matrix(embeddings)

    def _extract_concepts_and_entities(self, text):
        """
        Extract named entities and general concepts from one chunk.

        Parameters:
        -----------
        text : str
            The chunk text.

        Returns:
        --------
        concepts : list
            Normalized entities and concepts, deduplicated. Falls back to
            keywords when neither spaCy nor the LLM returns anything.
        """

        if text in self.concept_cache:
            return self.concept_cache[text]

        max_concepts = config.KNOWLEDGE_GRAPH_MAX_CONCEPTS
        named_entities = []

        if self.nlp is not None:
            document = self.nlp(text[: config.KNOWLEDGE_GRAPH_MAX_INPUT_CHARS])
            named_entities = [
                entity.text
                for entity in document.ents
                if entity.label_ in config.KNOWLEDGE_GRAPH_ENTITY_LABELS
            ]

        general_concepts = []
        if config.LLM_PROVIDER == "ollama":
            general_concepts = extract_concepts_with_ollama(text, max_concepts)

        if not named_entities and not general_concepts:
            general_concepts = extract_keywords(text, max_concepts)

        # Normalizing before deduplication is what makes shared-concept counts
        # meaningful: "Supreme Court" and "supreme court" are one concept.
        normalized = []
        for concept in named_entities + general_concepts:
            normalized_concept = normalize_concept(concept)
            if normalized_concept and normalized_concept not in normalized:
                normalized.append(normalized_concept)

        concepts = normalized[:max_concepts]
        self.concept_cache[text] = concepts

        return concepts

    def _extract_concepts(self, docs):
        """
        Extract concepts for every chunk, several chunks at a time.

        Parameters:
        -----------
        docs : list
            Normalized chunk dictionaries with 'id' and 'text'.

        Returns:
        --------
        None
        """

        with ThreadPoolExecutor(max_workers=config.KNOWLEDGE_GRAPH_CONCURRENCY) as executor:
            future_to_node = {
                executor.submit(self._extract_concepts_and_entities, doc.get("text", "")): str(doc["id"])
                for doc in docs
            }

            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    self.graph.nodes[node]["concepts"] = future.result()
                except Exception as exc:
                    logger.warning(f"Concept extraction failed for {node}: {exc}")
                    self.graph.nodes[node]["concepts"] = []

    def _add_edges(self, docs, embeddings, existing_ids=None, existing_embeddings=None):
        """
        Join chunks whose embeddings are similar enough to share an edge.

        The new chunks are compared against each other and, when the caller
        supplies them, against chunks already in the graph. Pairs of existing
        chunks are not recompared, since their edge was decided when they were
        ingested.

        Parameters:
        -----------
        docs : list
            Normalized chunk dictionaries with 'id'.
        embeddings : array-like
            Embeddings aligned with docs.
        existing_ids : list, optional
            Ids of chunks already in the graph.
        existing_embeddings : array-like, optional
            Embeddings aligned with existing_ids.

        Returns:
        --------
        None
        """

        node_ids = [str(doc["id"]) for doc in docs]
        similarity_matrix = self._compute_similarities(embeddings)

        if similarity_matrix.shape[0] != len(node_ids):
            raise ValueError("embeddings and chunks must be the same length")

        # Only the upper triangle: the graph is undirected and self loops are
        # meaningless here.
        rows, columns = np.triu_indices(len(node_ids), k=1)
        above_threshold = similarity_matrix[rows, columns] > self.edges_threshold

        for row, column in zip(rows[above_threshold], columns[above_threshold]):
            self._add_edge(node_ids[row],node_ids[column],float(similarity_matrix[row][column]))

        if not existing_ids:
            return

        existing_ids = [str(node_id) for node_id in existing_ids]
        cross_similarities = _cross_similarities(embeddings, existing_embeddings)

        if cross_similarities.shape != (len(node_ids), len(existing_ids)):
            raise ValueError("existing embeddings and ids must be the same length")

        rows, columns = np.nonzero(cross_similarities > self.edges_threshold)

        for row, column in zip(rows, columns):
            self._add_edge(node_ids[row],existing_ids[column],float(cross_similarities[row][column]))

    def _add_edge(self, node1, node2, similarity_score):
        """
        Add one weighted edge between two nodes already present in the graph.

        Parameters:
        -----------
        node1 : str
            The first node id.
        node2 : str
            The second node id.
        similarity_score : float
            The cosine similarity between the two chunks.

        Returns:
        --------
        None
        """

        if node1 == node2:
            return

        shared_concepts = set(self.graph.nodes[node1]["concepts"]) & set(
            self.graph.nodes[node2]["concepts"]
        )
        edge_weight = self._calculate_edge_weight(
            node1, node2, similarity_score, shared_concepts
        )

        self.graph.add_edge(
            node1,
            node2,
            weight=edge_weight,
            similarity=similarity_score,
            shared_concepts=sorted(shared_concepts),
        )

    def _calculate_edge_weight(
        self, node1, node2, similarity_score, shared_concepts, alpha=None, beta=None
    ):
        """
        Blend embedding similarity and shared concepts into an edge weight.

        Parameters:
        -----------
        node1 : str
            The first node id.
        node2 : str
            The second node id.
        similarity_score : float
            The cosine similarity between the two chunks.
        shared_concepts : set
            The concepts both chunks carry.
        alpha : float, optional
            The weight given to similarity. Defaults to
            config.KNOWLEDGE_GRAPH_SIMILARITY_WEIGHT.
        beta : float, optional
            The weight given to shared concepts. Defaults to
            config.KNOWLEDGE_GRAPH_CONCEPT_WEIGHT.

        Returns:
        --------
        weight : float
            The edge weight.
        """

        alpha = config.KNOWLEDGE_GRAPH_SIMILARITY_WEIGHT if alpha is None else alpha
        beta = config.KNOWLEDGE_GRAPH_CONCEPT_WEIGHT if beta is None else beta

        max_possible_shared = min(
            len(self.graph.nodes[node1]["concepts"]),
            len(self.graph.nodes[node2]["concepts"]),
        )
        normalized_shared_concepts = (
            len(shared_concepts) / max_possible_shared if max_possible_shared else 0.0
        )

        return alpha * similarity_score + beta * normalized_shared_concepts

    def related_chunk_ids(self, chunk_ids, hops=1, limit=None, with_weights=False):
        """
        Walk out from a set of chunks to the chunks most strongly linked to them.

        Parameters:
        -----------
        chunk_ids : iterable
            The chunk ids to start from, typically the ids a retriever returned.
        hops : int
            How many edges to follow out from the starting chunks.
        limit : int, optional
            The maximum number of neighbours to return.
        with_weights : bool
            Return (chunk_id, weight) pairs instead of bare ids, so a caller can
            rank or score the neighbours it pulled in.

        Returns:
        --------
        related : list
            Neighbouring chunk ids, strongest edge weight first, excluding the
            starting chunks.
        """

        seeds = {
            str(chunk_id) for chunk_id in chunk_ids if str(chunk_id) in self.graph
        }
        if not seeds:
            return []

        best_weight = {}
        frontier = set(seeds)

        for _ in range(max(1, hops)):
            next_frontier = set()

            for node in frontier:
                for neighbour, edge in self.graph[node].items():
                    if neighbour in seeds:
                        continue

                    weight = edge.get("weight", 0.0)
                    if weight > best_weight.get(neighbour, 0.0):
                        best_weight[neighbour] = weight
                        next_frontier.add(neighbour)

            frontier = next_frontier
            if not frontier:
                break

        related = sorted(best_weight, key=best_weight.get, reverse=True)
        related = related[:limit] if limit else related

        if with_weights:
            return [(node, best_weight[node]) for node in related]

        return related

    def to_dict(self):
        """
        Serialize the graph to a plain JSON friendly dictionary.

        Chunk text is not stored: it already lives in the FAISS metadata, and
        duplicating it would double the size of the graph file.

        Parameters:
        -----------
        None

        Returns:
        --------
        data : dict
            The serialized graph.
        """

        return {
            "version": 1,
            "edges_threshold": self.edges_threshold,
            "nodes": [
                {"id": node, "concepts": attributes.get("concepts", [])}
                for node, attributes in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": attributes.get("weight", 0.0),
                    "similarity": attributes.get("similarity", 0.0),
                    "shared_concepts": attributes.get("shared_concepts", []),
                }
                for source, target, attributes in self.graph.edges(data=True)
            ],
        }

    def save(self, graph_path=None):
        """
        Write the graph atomically, so a crash cannot leave partial JSON behind.

        Parameters:
        -----------
        graph_path : str, optional
            Defaults to config.KNOWLEDGE_GRAPH_PATH.

        Returns:
        --------
        None
        """

        graph_path = Path(graph_path or config.KNOWLEDGE_GRAPH_PATH)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = graph_path.with_suffix(graph_path.suffix + ".tmp")

        with temporary_path.open("w", encoding="utf-8") as graph_file:
            json.dump(self.to_dict(), graph_file, ensure_ascii=False)
            graph_file.flush()
            os.fsync(graph_file.fileno())

        os.replace(temporary_path, graph_path)
        logger.info(f"Saved knowledge graph to {graph_path}")

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a graph from its serialized form.

        Parameters:
        -----------
        data : dict
            A dictionary produced by to_dict.

        Returns:
        --------
        knowledge_graph : KnowledgeGraph
            The rebuilt graph.
        """

        knowledge_graph = cls(edges_threshold=data.get("edges_threshold"))

        for node in data.get("nodes", []):
            knowledge_graph.graph.add_node(
                node["id"], text="", concepts=node.get("concepts", [])
            )

        for edge in data.get("edges", []):
            knowledge_graph.graph.add_edge(
                edge["source"],
                edge["target"],
                weight=edge.get("weight", 0.0),
                similarity=edge.get("similarity", 0.0),
                shared_concepts=edge.get("shared_concepts", []),
            )

        return knowledge_graph

    @classmethod
    def load(cls, graph_path=None):
        """
        Load a saved graph from disk.

        Parameters:
        -----------
        graph_path : str, optional
            Defaults to config.KNOWLEDGE_GRAPH_PATH.

        Returns:
        --------
        knowledge_graph : KnowledgeGraph or None
            The loaded graph, or None when no graph file exists yet.
        """

        graph_path = Path(graph_path or config.KNOWLEDGE_GRAPH_PATH)

        if not graph_path.is_file():
            return None

        with graph_path.open("r", encoding="utf-8") as graph_file:
            return cls.from_dict(json.load(graph_file))

    def build_graph(self, docs, embeddings, existing_ids=None, existing_embeddings=None):
        """
        Build the graph by adding nodes, extracting concepts and adding edges.

        Parameters:
        -----------
        docs : list
            Normalized chunk dictionaries with 'id' and 'text'.
        embeddings : array-like
            Embeddings aligned with docs.
        existing_ids : list, optional
            Ids of chunks already in the graph that the new chunks should also
            be compared against, so edges form across ingestion runs.
        existing_embeddings : array-like, optional
            Embeddings aligned with existing_ids.

        Returns:
        --------
        self : KnowledgeGraph
            The populated graph, for chaining.
        """

        if not docs:
            logger.info("No chunks supplied; knowledge graph left unchanged")
            return self

        self._add_nodes(docs)

        self._extract_concepts(docs)
        self._add_edges(docs,embeddings,existing_ids=existing_ids,existing_embeddings=existing_embeddings)

        logger.info(
            f"Knowledge graph now holds {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

        return self

def build_knowledge_graph(docs, embeddings, graph_path=None, commit=True, store=None, link_to_existing=True):
    """
    Add chunks to the saved knowledge graph, extending it if one already exists.

    Parameters:
    -----------
    docs : list
        Normalized chunk dictionaries with 'id' and 'text'.
    embeddings : array-like
        Embeddings aligned with docs, reused from the ingestion pipeline so the
        chunks are not embedded twice.
    graph_path : str, optional
        Defaults to config.KNOWLEDGE_GRAPH_PATH.
    commit : bool
        Pass False to build in memory only, when the caller publishes the graph
        alongside the other indexes.
    store : FAISSStore, optional
        The store holding the vectors of previously ingested chunks.
    link_to_existing : bool
        Whether the new chunks should also be compared against chunks already
        in the graph, which is what produces edges across documents.

    Returns:
    --------
    knowledge_graph : KnowledgeGraph
        The extended graph.
    """

    graph_path = graph_path or config.KNOWLEDGE_GRAPH_PATH
    knowledge_graph = KnowledgeGraph.load(graph_path) or KnowledgeGraph()

    existing_ids = []
    existing_embeddings = None

    if link_to_existing:
        new_ids = {str(doc["id"]) for doc in docs}
        previous_ids = [node for node in knowledge_graph.graph if node not in new_ids]
        existing_ids, existing_embeddings = load_store_embeddings(previous_ids, store)

        if previous_ids and not existing_ids:
            logger.warning(
                "No stored vectors found for existing graph nodes; "
                "this batch will only link to itself"
            )

    knowledge_graph.build_graph(
        docs,
        embeddings=embeddings,
        existing_ids=existing_ids,
        existing_embeddings=existing_embeddings,
    )

    if commit:
        knowledge_graph.save(graph_path)

    return knowledge_graph
