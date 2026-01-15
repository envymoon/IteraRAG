import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def expand_context(retrieved_chunks, pubmed_map, window_size=1):
    expanded_results = []
    for hit in retrieved_chunks:
        meta = hit['metadata']
        if meta.get('source') != 'PubMed':
            expanded_results.append(hit['content'])
            continue

        abs_id = meta['abstract_id']
        curr_line = meta['line_number']
        all_lines = pubmed_map.get(abs_id, [])
        neighbors = [
            c['content'] for c in all_lines
            if abs(c['metadata']['line_number'] - curr_line) <= window_size
        ]
        expanded_results.append(" ".join(neighbors))
    return expanded_results


def rrf_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, idx in enumerate(vector_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    for rank, idx in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    def __init__(self, final_chunks, embeddings):
        self.final_chunks = final_chunks
        self.text_list = [c['content'] for c in final_chunks]
        self.embeddings = np.array(embeddings).astype("float32")

        d = self.embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(self.embeddings)

        tokenized_corpus = [c.lower().split() for c in self.text_list]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

        self.target_to_indices = {}
        for i, c in enumerate(final_chunks):
            t = c['metadata'].get("target")
            if t:
                self.target_to_indices.setdefault(t, []).append(i)

    def search(self, user_query, query_vec, pubmed_map, k=30, target_filter=None, rerank_top_n=5):
        allowed_indices = self.target_to_indices.get(target_filter) if target_filter else None

        self.index.hnsw.efSearch = 128
        if allowed_indices:
            selector = faiss.IDSelectorBatch(allowed_indices)
            params = faiss.SearchParameters(sel=selector)
            _, I_vector = self.index.search(query_vec, k*2, params=params)
        else:
            _, I_vector = self.index.search(query_vec, k*2)
        vector_indices = I_vector[0].tolist()

        tokenized_query = user_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        if allowed_indices:
            mask = np.zeros_like(bm25_scores)
            mask[allowed_indices] = 1
            bm25_scores *= mask
        bm25_indices = np.argsort(bm25_scores)[::-1][:k*2].tolist()

        combined = rrf_fusion(vector_indices, bm25_indices)

        candidates = [self.final_chunks[i] for i in combined[:10]]
        pairs = [[user_query, c['content']] for c in candidates]
        scores = self.reranker.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        top_chunks = reranked[:rerank_top_n]
        expanded = expand_context(top_chunks, pubmed_map, window_size=1)

        return expanded
