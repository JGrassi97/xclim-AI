# xclim_tools.rag.utils
# =================================================

# Helper functions for embedding retrieval and similarity search.
# Used in xclim RAG pipelines for retrieving semantically relevant tools.

import numpy as np
from typing import Any, List, Optional, Tuple


def safe_get_embeddings(
    store: Any,
    batch_size: int = 1,
    valid_tools_names: Optional[List[str]] = None
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Safely retrieve all embeddings from the vector store.

    Parameters
    ----------
    store : Any
        Vector store with a `_collection` attribute implementing `.get(...)`.
    batch_size : int
        Number of document IDs to query per batch.
    valid_tools_names : list of str, optional
        If provided, only keep embeddings whose `id` field matches one of these.

    Returns
    -------
    ids : list of str
        IDs of the retrieved documents.
    embeddings : list of ndarray
        Corresponding embedding vectors.
    """
    ids = store._collection.get()["ids"]
    all_embeddings = []
    all_ids = []

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        try:
            res = store._collection.get(ids=batch_ids, include=["embeddings", "metadatas"])
            embeddings = res["embeddings"]
            metadatas = res["metadatas"]
            ids_batch = res["ids"]

            for emb, meta, id_ in zip(embeddings, metadatas, ids_batch):
                if valid_tools_names is None:
                    all_embeddings.append(emb)
                    all_ids.append(id_)
                else:
                    tool_id = meta.get("id", "").split(".")[-1]
                    if tool_id in valid_tools_names:
                        all_embeddings.append(emb)
                        all_ids.append(id_)

        except Exception as e:
            print(f"Error in batch {i // batch_size + 1} (IDs {i} to {i + batch_size}): {e}")
            continue

    return all_ids, all_embeddings


def get_top_k(
    similarity_matrix: np.ndarray,
    ids: List[str],
    store: Any,
    k: int = 5
) -> List[dict]:
    """
    Retrieve the top-k most similar documents from a similarity matrix.

    Parameters
    ----------
    similarity_matrix : ndarray
        2D similarity matrix (typically shape (1, N)).
    ids : list of str
        IDs corresponding to the matrix columns.
    store : Any
        Vector store with a `.get(...)` method.
    k : int
        Number of top results to return.

    Returns
    -------
    results : list of dict
        Each dict contains rank, id, text, and similarity score.
    """
    topk_idx = np.argsort(similarity_matrix[0])[-k:][::-1]
    results = []

    for rank, i in enumerate(topk_idx, 1):
        result = store.get([ids[i]])
        meta_id = result["metadatas"][0].get("id", "N/A")
        text = result["documents"][0]

        results.append({
            "rank": rank,
            "id": meta_id,
            "text": text,
            "similarity": similarity_matrix[0][i],
        })

    return results