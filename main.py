import os
import json
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm

from litellm import completion
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Keyphrase(BaseModel):
    keywords: list[str]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

def hungarian_similarity(predicted_embeddings, ground_truth_embeddings, penalty_cost=100.0):
    """
    Computes the optimal matching between predicted and ground truth embeddings using the Hungarian algorithm.
    Pads the cost matrix with a high penalty for unmatched items.
    """
    # Compute the pairwise cosine similarity matrix.
    similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
    cost_matrix = -similarity_matrix  # Higher similarity → lower cost.
    
    N, M = cost_matrix.shape
    size = max(N, M)
    padded_cost = np.full((size, size), penalty_cost, dtype=float)
    padded_cost[:N, :M] = cost_matrix
    
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Filter out dummy matches.
    valid_matches = [(i, j, similarity_matrix[i, j])
                     for i, j in zip(row_ind, col_ind) if i < N and j < M]
    matched_similarities = np.array([sim for (_, _, sim) in valid_matches])
    avg_similarity = np.mean(matched_similarities) if matched_similarities.size > 0 else 0.0
    
    return row_ind, col_ind, matched_similarities, avg_similarity

base_prompt = """
Given the following text, extract the most relevant keywords or keyphrases that best summarize its content.
Try to minimize overlap of concepts in keyphrases.

Return the keywords as a comma-separated list of phrases.
Text:
{}
Keywords:
"""

def bench_sample(llm, sample):
    doc_text = " ".join(sample["document"])
    gt_keyphrases = sample["extractive_keyphrases"] + sample["abstractive_keyphrases"]
    
    response = completion(
        model=llm,
        messages=[{"role": "user", "content": base_prompt.format(doc_text)}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "keyphrase_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "keyphrases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of extracted keyphrases"
                        }
                    },
                    "required": ["keyphrases"],
                    "additionalProperties": False
                }
            }
        }
    )
    
    generated_keyphrases = json.loads(response.choices[0].message.content)["keyphrases"]
    
    generated_embeddings = [
        embedding_model.encode(keyphrase, normalize_embeddings=True)
        for keyphrase in generated_keyphrases
    ]
    gt_embeddings = [
        embedding_model.encode(keyphrase, normalize_embeddings=True)
        for keyphrase in gt_keyphrases
    ]
    
    similarity = hungarian_similarity(generated_embeddings, gt_embeddings)
    
    return {
        "generated_keyphrases": generated_keyphrases, 
        "ground_truth_keyphrases": gt_keyphrases, 
        "hungarian_similarity": similarity
    }

dataset = load_dataset("midas/krapivin", "raw")["test"]

llm_model = "openrouter/google/gemini-2.0-flash-001"

def process_sample(sample):
    try:
        return bench_sample(llm_model, sample)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()-2) as executor:
    futures = [executor.submit(process_sample, sample) for sample in dataset]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
        result = future.result()
        if result is not None:
            results.append(result)

all_avg_sim = np.mean([res["hungarian_similarity"][-1] for res in results])
print(f"Overall average similarity across dataset: {all_avg_sim}")
