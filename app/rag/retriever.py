import heapq
import json
from pathlib import Path

import numpy as np
import faiss

from app import config
from app.ingestion.utils.faiss import FAISSStore
from app.ingestion.utils.knowledge_graph_methods import KnowledgeGraph
from app.logger import Logger
from app.rag.utils.reranking_methods import (
    greedy_dartboard_search,
    weighted_reciprocal_rank_fusion,
)
from app.rag.utils.retrieval_methods import (
    bm25_retrieval,
    get_best_segments,
    hierarchical_retrieval,
    hyde_retrieval
)
retrieval_methods = {"bm25_retrieval": bm25_retrieval,"get_best_segments": get_best_segments,
"hierarchical_retrieval": hierarchical_retrieval,"hyde_retrieval": hyde_retrieval}

reranking_methods = {"greedy_dartboard_search": greedy_dartboard_search,
"weighted_reciprocal_rank_fusion": weighted_reciprocal_rank_fusion}



logger = Logger.get_logger(__name__)

def _dartboard_rerank(vectorstore, query_vector, documents, k):
    """Rerank retrieved candidates for relevance and vector-space diversity."""
    if not documents or k <= 0:
        return []

    metadata_indices = {
        document.get("id"): index
        for index, document in enumerate(vectorstore.metadata)
        if document.get("id") is not None
    }
    candidates = [
        document for document in documents if document.get("id") in metadata_indices
    ]
    if not candidates:
        return documents[:k]

    candidate_vectors = np.asarray(
        [
            vectorstore.index.reconstruct(metadata_indices[document["id"]])
            for document in candidates
        ],
        dtype=np.float32,
    )
    vector_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    candidate_vectors = candidate_vectors / np.maximum(vector_norms, 1e-12)

    query_norm = max(float(np.linalg.norm(query_vector)), 1e-12)
    query_vector = query_vector / query_norm

    query_distances = np.maximum(
        0.0,
        1.0 - candidate_vectors @ query_vector,
    )
    document_distances = np.maximum(
        0.0,
        1.0 - candidate_vectors @ candidate_vectors.T,
    )
    selected_documents, selection_scores = greedy_dartboard_search(
        query_distances,
        document_distances,
        candidates,
        k,
    )
    return [
        {**document, "relevance_score": score}
        for document, score in zip(selected_documents, selection_scores)
    ]


# The graph is reread only when ingestion has rewritten it, so a query does not
# pay for parsing the whole file every time.
_graph_cache = {"mtime": None, "graph": None}


def load_knowledge_graph(graph_path=None):
    """Load the saved knowledge graph, reusing the cached copy until it changes.

    parameters:
    -----------
    graph_path : str, optional
        Defaults to config.KNOWLEDGE_GRAPH_PATH.

    returns:
    --------
    knowledge_graph : KnowledgeGraph or None
        The graph, or None when no graph has been built yet.
    """

    graph_path = Path(graph_path or config.KNOWLEDGE_GRAPH_PATH)

    if not graph_path.is_file():
        return None

    modified_at = graph_path.stat().st_mtime

    if _graph_cache["graph"] is None or _graph_cache["mtime"] != modified_at:
        _graph_cache["graph"] = KnowledgeGraph.load(graph_path)
        _graph_cache["mtime"] = modified_at
        logger.info(f"Loaded knowledge graph with {_graph_cache['graph'].graph.number_of_nodes()} nodes")

    return _graph_cache["graph"]


def _metadata_by_id(store=None):
    """Map chunk id to its stored chunk, for looking up expanded neighbours.

    parameters:
    -----------
    store : FAISSStore, optional
        An open store. Metadata is read from disk when omitted.

    returns:
    --------
    metadata_by_id : dict
        Chunk id to chunk dictionary.
    """

    if store is not None:
        return {chunk["id"]: chunk for chunk in store.metadata if "id" in chunk}

    with open(config.METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        return {chunk["id"]: chunk for chunk in json.load(metadata_file) if "id" in chunk}


def context_answers_question(question, context):
    """Ask the model whether the context gathered so far already answers the question.

    This is the traversal's early stop: once the accumulated chunks contain the
    answer, walking further out only adds noise.

    parameters:
    -----------
    question : str
        The user's question.
    context : str
        The text of every chunk gathered so far.

    returns:
    --------
    is_complete : bool
        True when the model reports the context is sufficient. False whenever the
        model is unreachable or answers unclearly, so an unavailable model makes
        the traversal run to its budget rather than stopping early.
    """

    import requests

    prompt = f"""You are a legal assistant judging whether you have enough context.

Given the question and the context below, does the context provide a complete
answer to the question?

Respond with JSON only, in the form {{"is_complete": true, "answer": "..."}}.

QUESTION
{question}

CONTEXT
{context[: config.KNOWLEDGE_GRAPH_MAX_CONTEXT_CHARS]}"""

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
        payload = json.loads(response.json().get("response", "{}"))
    except Exception as exc:
        logger.warning(f"Answer check failed, continuing traversal: {exc}")
        return False

    is_complete = bool(payload.get("is_complete"))

    if is_complete:
        logger.info(f"Context answers the question after traversal: {str(payload.get('answer'))[:200]}")

    return is_complete


def graph_traversal(question, results, store=None, metadata_by_id=None, max_nodes=None):
    """Walk the knowledge graph out from the retrieved chunks, strongest links first.

    Retrieval scores each chunk against the query alone, so a chunk that supplies
    the context for a match (the section a passage relies on, the same doctrine
    argued in another judgment) stays invisible unless it happens to match the
    query too. This is a Dijkstra-style walk over the graph to reach those chunks:
    every retrieved chunk is a source at distance zero, and following an edge costs
    1 / weight, so strong links are traversed before weak ones and a chain of
    strong links outranks a single weak one.

    Two things keep the walk from wandering. A node whose concepts have all been
    seen already is not expanded, since it leads back over ground the traversal has
    covered; and the walk stops once it hits the node or character budget, or once
    the model reports the gathered context answers the question.

    parameters:
    -----------
    question : str
        The user's question, used only for the optional answer check.
    results : list
        The chunks retrieval returned, each with an 'id'.
    store : FAISSStore, optional
        An open store, used to look up the text of the traversed chunks.
    metadata_by_id : dict, optional
        A chunk lookup the caller has already built.
    max_nodes : int, optional
        The most chunks the traversal may add. Defaults to
        config.KNOWLEDGE_GRAPH_EXPANSION_K.

    returns:
    --------
    expanded_results : list
        The original results in their retrieval order, followed by the traversed
        chunks in the order they were reached.
    """

    if not results:
        return results

    knowledge_graph = load_knowledge_graph()

    if knowledge_graph is None:
        logger.info("No knowledge graph on disk; returning retrieval results unchanged")
        return results

    graph = knowledge_graph.graph
    seed_ids = [chunk["id"] for chunk in results if chunk.get("id") in graph]

    if not seed_ids:
        logger.info("None of the retrieved chunks are in the knowledge graph")
        return results

    if metadata_by_id is None:
        metadata_by_id = _metadata_by_id(store)

    max_nodes = config.KNOWLEDGE_GRAPH_EXPANSION_K if max_nodes is None else max_nodes
    seeds = set(seed_ids)

    # Multi-source Dijkstra: every retrieved chunk starts at distance zero, and
    # retrieval rank breaks ties so the best chunk's neighbourhood is explored
    # first. The rank travels with the walk to keep the ordering deterministic.
    queue = [(0.0, rank, node) for rank, node in enumerate(seed_ids)]
    heapq.heapify(queue)
    distances = {node: 0.0 for node in seed_ids}

    visited_concepts = set()
    traversal_path = []
    collected = []
    context_length = sum(len(chunk.get("text", "")) for chunk in results)

    while queue:
        distance, rank, node = heapq.heappop(queue)

        # A shorter route to this node was queued after this entry.
        if distance > distances.get(node, float("inf")):
            continue

        traversal_path.append(node)

        if node not in seeds and node in metadata_by_id:
            chunk = metadata_by_id[node]
            collected.append({
                **chunk,
                "graph_distance": round(float(distance), 4),
                "graph_step": len(collected) + 1,
                "retrieval_source": "knowledge_graph",
            })
            context_length += len(chunk.get("text", ""))

            if len(collected) >= max_nodes or context_length >= config.KNOWLEDGE_GRAPH_MAX_CONTEXT_CHARS:
                break

            if config.KNOWLEDGE_GRAPH_ANSWER_CHECK:
                gathered = "\n\n".join(chunk.get("text", "") for chunk in results + collected)
                if context_answers_question(question, gathered):
                    break

        # Expand only when this node brought concepts the walk has not seen; a
        # node that adds nothing new leads back into ground already covered.
        concepts = set(graph.nodes[node].get("concepts", []))
        introduces_new_concepts = not concepts or not concepts.issubset(visited_concepts)
        visited_concepts |= concepts

        if not introduces_new_concepts:
            continue

        for neighbour, edge in graph[node].items():
            weight = edge.get("weight", 0.0)

            if weight <= 0:
                continue

            # Higher weight means a stronger link, so it has to cost less to cross.
            neighbour_distance = distance + (1.0 / weight)

            if neighbour_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = neighbour_distance
                heapq.heappush(queue, (neighbour_distance, rank, neighbour))

    logger.info(f"Knowledge graph traversal visited {len(traversal_path)} nodes and added {len(collected)} chunks")

    return list(results) + collected


def retrieve(question, store=None, k=5):
    """Retrieve documents using the method selected in app.config."""

    try:
        from app.ingestion.embedder import embed_batch
        retrieval_method = retrieval_methods.get(config.RETRIEVAL_METHOD)
        metadata_by_id = None

        if config.RETRIEVAL_METHOD == "hierarchical_retrieval":
            summary_store = FAISSStore.load(config.SUMMARY_VECTOR_STORE_PATH, config.SUMMARY_METADATA_PATH)    
            results = retrieval_method(summary_store, store, question, k=k, summary_k=config.HIERARCHICAL_SUMMARY_K)
        
        elif config.RETRIEVAL_METHOD == "fusion_retrieval":
            with open(config.BM25_INDEX_PATH, "r", encoding="utf-8") as index_file:
                bm25_index = json.load(index_file)

            bm25_results = bm25_retrieval(question, bm25_index, k)
            logger.info(f"BM25 results: {bm25_results}")


            index = faiss.read_index(config.VECTOR_STORE_PATH)
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata_by_id = {chunk["id"]: chunk for chunk in metadata}

            query_vector = embed_batch([question])
            search_k = min(k, index.ntotal)
            vector_scores, indices = index.search(query_vector, search_k)
            vector_results = [
                {
                    "chunk_id": metadata[chunk_index]["id"],
                    "vector_score": float(vector_score),
                }
                for vector_score, chunk_index in zip(vector_scores[0], indices[0])
                if chunk_index >= 0
            ]
            logger.info(f"Vector results: {vector_results}")
            
        elif config.RETRIEVAL_METHOD == "hyde_retrieval":
            vector_results = retrieval_method(question, store, k=k, chunk_size=config.CHUNK_SIZE)


        if config.RERANKING_METHOD == "greedy_dartboard_search":
            query_vector = embed_batch([question])
            results = _dartboard_rerank(store, query_vector, vector_results, k)
        elif config.RERANKING_METHOD == "weighted_reciprocal_rank_fusion":
            weights = {"bm25": config.FUSION_ALPHA, "vector": 1 - config.FUSION_ALPHA}
            chunks_to_return = weighted_reciprocal_rank_fusion(
                bm25_results, vector_results, weights, id_key="chunk_id"
            )
            results = [
                {
                    **metadata_by_id[chunk_info["id"]],     #unpacking dictionary to include all metadata_by_id fields
                    "rrf_score": chunk_info["rrf_score"],
                    "final_rank": chunk_info["final_rank"],
                }
                for chunk_info in chunks_to_return if chunk_info["id"] in metadata_by_id
            ]
            logger.info(f"RRF results: {chunks_to_return}")
            logger.info(f"Final results: {results}")

        if config.USE_KNOWLEDGE_GRAPH_EXPANSION:
            try:
                results = graph_traversal(question, results, store=store, metadata_by_id=metadata_by_id)
            except Exception as exc:
                # Expansion is additive; a broken graph must not cost us the
                # chunks retrieval already found.
                logger.warning(f"Knowledge graph traversal skipped: {exc}")

    except Exception as exc:
        logger.error(f"Error retrieving documents: {exc}")
        return []

    return results
