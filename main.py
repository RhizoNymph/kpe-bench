import os
import json
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count
from tqdm import tqdm

from litellm import completion, token_counter
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
    
    prompt = base_prompt.format(doc_text)
    prompt_tokens = token_counter(model=llm, messages=[{"role": "user", "content": prompt}])
    
    response = completion(
        model=llm,
        messages=[{"role": "user", "content": prompt}],
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
    
    completion_tokens = response.usage.completion_tokens
    
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
        "hungarian_similarity": similarity,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }

dataset = load_dataset("midas/krapivin", "raw")["test"]

llm_model = "openrouter/anthropic/claude-3.5-haiku-20241022:beta"

def process_sample(sample):
    try:
        return bench_sample(llm_model, sample)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def estimate_dataset_tokens(llm_model, dataset):
    """
    Estimate total tokens that will be used across the dataset.
    - Counts exact prompt tokens for all documents
    - Estimates completion tokens based on first 10 samples
    """
    # Count exact prompt tokens for all documents
    print(f"Counting exact prompt tokens for all {len(dataset)} documents...")
    total_prompt_tokens = 0
    for i, sample in enumerate(tqdm(dataset, desc="Counting prompt tokens")):
        doc_text = " ".join(sample["document"])
        prompt = base_prompt.format(doc_text)
        prompt_tokens = token_counter(model=llm_model, messages=[{"role": "user", "content": prompt}])
        total_prompt_tokens += prompt_tokens
    
    # Run completions on first 10 samples to estimate completion tokens
    first_n = min(10, len(dataset))
    print(f"Running completions on first {first_n} samples to estimate completion tokens...")
    total_completion_sample_tokens = 0
    
    for i in tqdm(range(first_n), desc="Processing samples for completion token estimation"):
        sample = dataset[i]
        doc_text = " ".join(sample["document"])
        gt_keyphrases = sample["extractive_keyphrases"] + sample["abstractive_keyphrases"]
        
        prompt = base_prompt.format(doc_text)
        
        try:
            response = completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
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
            completion_tokens = response.usage.completion_tokens
            total_completion_sample_tokens += completion_tokens
        except Exception as e:
            print(f"Error estimating completion tokens for sample {i}: {e}")
            continue
    
    # Extrapolate completion tokens for the full dataset
    avg_completion_tokens = total_completion_sample_tokens / first_n if first_n > 0 else 0
    estimated_total_completion_tokens = avg_completion_tokens * len(dataset)
    
    return {
        "prompt_tokens": total_prompt_tokens,
        "estimated_completion_tokens": int(estimated_total_completion_tokens),
        "estimated_total_tokens": int(total_prompt_tokens + estimated_total_completion_tokens)
    }

# Update the print statements
token_estimates = estimate_dataset_tokens(llm_model, dataset)
print(f"Estimated token usage for entire dataset:")
print(f"  - Prompt tokens (exact): {token_estimates['prompt_tokens']:,}")
print(f"  - Completion tokens (estimated): {token_estimates['estimated_completion_tokens']:,}")
print(f"  - Total tokens: {token_estimates['estimated_total_tokens']:,}")

# Ask for confirmation before proceeding
proceed = input("Do you want to proceed with processing the dataset? (y/n): ")
if proceed.lower() != 'y':
    print("Operation cancelled by user.")
    exit()

total_prompt_tokens = 0
total_completion_tokens = 0
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()-2) as executor:
    futures = [executor.submit(process_sample, sample) for sample in dataset]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing samples"):
        result = future.result()
        if result is not None:
            results.append(result)
            total_prompt_tokens += result["prompt_tokens"]
            total_completion_tokens += result["completion_tokens"]

all_avg_sim = np.mean([res["hungarian_similarity"][-1] for res in results])
print(f"Overall average similarity across dataset: {all_avg_sim}")
print(f"Total tokens sent (prompts): {total_prompt_tokens}")
print(f"Total tokens received (completions): {total_completion_tokens}")
print(f"Total tokens overall: {total_prompt_tokens + total_completion_tokens}")
