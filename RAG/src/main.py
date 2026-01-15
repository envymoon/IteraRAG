import json, csv
import numpy as np
import torch
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from core.processor import (
    build_pubmed_chunks, load_txt_folder, build_pubmed_map
)
from core.retriever import HybridRetriever
from core.llm import load_llm, generate_answer


pubmed_chunks = build_pubmed_chunks("../data/raw/PubMed/train.csv")
libc_chunks = load_txt_folder("../data/raw/libc", "libc")
pytorch_chunks = load_txt_folder("../data/raw/pytorch", "pytorch")
final_chunks = libc_chunks + pytorch_chunks + pubmed_chunks

pubmed_map = build_pubmed_map(final_chunks)

# Embeddings 
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
text_list = [c['content'] for c in final_chunks]
embeddings = embedder.embed_documents(text_list)

retriever = HybridRetriever(final_chunks, embeddings)

# Load LLM
tokenizer, model = load_llm()

# Load Questions
questions = []
with open('../data/raw/questions.txt', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row.get('Question ', row.get('Question')).strip()
        gt = row.get('Ground Truth', row.get('ground_truth')).strip()
        if q:
            questions.append({"question": q, "ground_truth": gt})

# Run RAG 
results = []
for item in tqdm(questions):
    q = item["question"]
    gt = item["ground_truth"]

    q_vec = embedder.embed_query(q)
    q_vec = np.array([q_vec]).astype("float32")

    contexts = retriever.search(q, q_vec, pubmed_map, k=30)
    context_str = "\n\n".join(contexts)

    answer = generate_answer(model, tokenizer, context_str, q)

    results.append({
        "question": q,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": gt
    })

# Save
with open("../data/eval/results_final.json", "w") as f:
    json.dump(results, f, indent=2)

del model
torch.cuda.empty_cache()
