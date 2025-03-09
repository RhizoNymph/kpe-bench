import json

import numpy as np

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
    This function pads the cost matrix with a high penalty value for unmatched (dummy) entries.
    
    Args:
        predicted_embeddings (np.ndarray): Array of shape (N, D) for predicted keyphrase embeddings.
        ground_truth_embeddings (np.ndarray): Array of shape (M, D) for ground truth keyphrase embeddings.
        penalty_cost (float): Cost assigned to dummy entries.
        
    Returns:
        row_ind (np.ndarray): Indices of predicted embeddings that were matched.
        col_ind (np.ndarray): Indices of ground truth embeddings that were matched.
        matched_similarities (np.ndarray): Cosine similarity scores for the valid matches.
        avg_similarity (float): Average cosine similarity over valid matches.
    """
    # Compute the pairwise cosine similarity matrix (shape: [N, M])
    similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
    
    # Convert similarity matrix to cost matrix for minimization
    cost_matrix = -similarity_matrix
    
    # Determine dimensions and pad to a square matrix
    N, M = cost_matrix.shape
    size = max(N, M)
    padded_cost = np.full((size, size), penalty_cost, dtype=float)
    padded_cost[:N, :M] = cost_matrix
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    # Collect only valid matches (those within the original matrix bounds)
    valid_matches = []
    for i, j in zip(row_ind, col_ind):
        if i < N and j < M:
            valid_matches.append((i, j, similarity_matrix[i, j]))
    
    # Extract matched similarities and compute average
    matched_similarities = np.array([match[2] for match in valid_matches])
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

test_sample = dataset[0]

result = bench_sample("openrouter/google/gemini-2.0-flash-001", test_sample)

print(f"Generated keyphrases: {result['generated_keyphrases']}")
print(f"Ground truth keyphrases: {result['ground_truth_keyphrases']}")

rows, cols, matched_sim, avg_sim = result["hungarian_similarity"]
print(f"Average similarity: {avg_sim}")
