import json
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding Model for Evaluation
test_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return test_embeddings.embed_documents(texts)

# Metrics
def recall_at_k(contexts, gt, threshold=0.75):
    gt_emb = embed(gt)[0]
    ctx_embs = embed(contexts)
    sims = cosine_similarity([gt_emb], ctx_embs)[0]
    hit = np.max(sims) > threshold
    best_rank = int(np.argmax(sims)) + 1
    return hit, best_rank, float(np.max(sims))

def answer_gt_similarity(ans, gt):
    emb = embed([ans, gt])
    sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return float(sim)

def answer_context_similarity(ans, contexts):
    ans_emb = embed(ans)[0]
    ctx_embs = embed(contexts)
    sims = cosine_similarity([ans_emb], ctx_embs)[0]
    return float(np.max(sims))

def hallucination_flag(ans_gt_sim, ans_ctx_sim,
                       gt_th=0.6, ctx_th=0.6):
    return ans_gt_sim < gt_th and ans_ctx_sim < ctx_th

# Run Evaluation

def run_eval(result_path, output_path):
    with open(result_path) as f:
        results = json.load(f)

    eval_results = []

    for item in tqdm(results):
        q = item["question"]
        ans = item["answer"]
        contexts = item["contexts"]
        gt = item["ground_truth"]

        hit, rank, ctx_gt_sim = recall_at_k(contexts, gt)
        ans_gt_sim = answer_gt_similarity(ans, gt)
        ans_ctx_sim = answer_context_similarity(ans, contexts)
        hallucinated = hallucination_flag(ans_gt_sim, ans_ctx_sim)

        eval_results.append({
            "question": q,
            "hit": hit,
            "best_rank": rank,
            "ctx_gt_sim": ctx_gt_sim,
            "ans_gt_sim": ans_gt_sim,
            "ans_ctx_sim": ans_ctx_sim,
            "hallucinated": hallucinated
        })

    df = pd.DataFrame(eval_results)

    summary = {
        "Recall@k": float(df["hit"].mean()),
        "MRR": float((1 / df["best_rank"]).mean()),
        "Avg Context-GT Sim": float(df["ctx_gt_sim"].mean()),
        "Avg Answer-GT Sim": float(df["ans_gt_sim"].mean()),
        "Avg Answer-Context Sim": float(df["ans_ctx_sim"].mean()),
        "Hallucination Rate": float(df["hallucinated"].mean())
    }

    print("Evaluation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    run_eval(
        result_path="../data/eval/results.json",
        output_path="../data/eval/summary.json"
    )