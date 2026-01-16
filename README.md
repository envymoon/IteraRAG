# Multi-Stage Evolutionary RAG System & Retrieval Pipeline Optimization

## 1. Project Overview
This project aims to identify the most efficient and reliable **RAG (Retrieval-Augmented Generation)** configuration for a specific multi-source hybrid dataset. Instead of a static implementation, we explored the technological frontier through four iterative versions, using a custom quantitative scoring evaluation framework.

## 2. Dataset
* **Total Size:** ~420MB hybrid dataset
    * **PubMed:** 400MB of medical abstracts.
    * **Technical Docs:** 20MB from `libc` and `PyTorch` documentation.

## 3. Tech Stack
* **Embedding Model:** `BAAI/bge-small-en-v1.5`
* **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit quantization for local inference)
* **Vector Database/Search:** FAISS (IndexFlatL2, HNSW)
* **Retrieval Techniques:** BM25, Hybrid Search, RRF Fusion, Cross-Encoder Re-ranking

## 4. The Evolutionary Path (Four Versions)

| Version | Key Features & Architecture |
| :--- | :--- |
| **v1 (Baseline)** | Basic text chunking; **IndexFlatL2** exhaustive search. |
| **v2 (Advanced)** | Implemented **HNSW** for high-speed indexing; introduced **Hybrid Search** (Dense + BM25); utilized **RRF (Reciprocal Rank Fusion)** and metadata filtering for PubMed. |
| **v3 (Optimal)** | Built upon v2 with a **Cross-Encoder Re-ranker** for top-k refinement. Selected as the **final version** due to stability and speed. |
| **v4 (Experimental)** | Optimized chunking for technical docs; applied metadata mapping and context expansion across all sources. |

## 5. Quantitative Evaluation
Due to restricted API access for tools like RAGAS, this project uses a custom suite to quantify performance:

### Core Definitions
* $G$: Ground Truth text
* $A$: Generated Answer text
* $C = \{c_1, c_2, \dots, c_k\}$: Set of top-$k$ retrieved context chunks
* $E(x)$: Embedding vector of text $x$
* $sim(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$: Cosine similarity between two vectors

### Metrics & Formulas

#### 1. Recall@k
Determines if the required information is captured within the top-k retrieved chunks.
$$\text{Recall@k} = \mathbb{1} \left( \max_{c \in C} \left[ sim(E(c), E(G)) \right] \geq \tau_{recall} \right)$$
*(Threshold $\tau_{recall} = 0.75$)*

#### 2. Answer-GT Similarity
Measures the semantic alignment between the LLM response and the reference answer.
$$\text{Sim}(A, G) = \frac{E(A) \cdot E(G)}{\|E(A)\| \|E(G)\|}$$

#### 3. Answer-Context Similarity (Grounding)
Ensures the answer is derived from the provided documents.
$$\text{Sim}(A, C) = sim(E(A), E(C_{concat}))$$

#### 4. Hallucination Rate
A heuristic flag when the answer lacks similarity to both the ground truth and context.
**Individual Flag ($H_i$):**
$$H_i = \begin{cases} 1, & \text{if } \text{Sim}(A, G) < \tau_{h} \text{ AND } \text{Sim}(A, C) < \tau_{h} \\ 0, & \text{otherwise} \end{cases}$$

**Overall Rate:**
$$\text{Hallucination Rate} = \frac{1}{N} \sum_{i=1}^{N} H_i$$