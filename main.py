import os
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

def hungarian_similarity(predicted_embeddings, ground_truth_embeddings):
    # Compute the pairwise cosine similarity matrix (shape: [N, M])
    similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
    
    # Convert similarity matrix to a cost matrix for the minimization problem.
    # We use the negative because a higher similarity should have a lower cost.
    cost_matrix = -similarity_matrix
    
    # Solve the linear assignment problem using the Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Extract the similarity scores for the optimal matching.
    matched_similarities = similarity_matrix[row_ind, col_ind]
    
    # Compute an average similarity score (this can be used as an evaluation metric).
    avg_similarity = np.mean(matched_similarities) if len(matched_similarities) > 0 else 0.0
    
    return row_ind, col_ind, matched_similarities, avg_similarity

base_prompt = """
    Given the following text, extract the most relevant keywords or keyphrases that best summarize its content.  
    Try to minimze overlap of concepts in keyphrases.

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

    generated_embeddings = [embedding_model.encode(generated_keyphrase, normalize_embeddings=True) for generated_keyphrase in ]
    gt_embeddings = embedding_model(gt_keyphrases, normalize_embeddings=True)

    similarity = hungarian_similarity(generated_embeddings, gt_embeddings)

    return {
            "generated keyphrases": generated_keyphrases, 
            "ground truth keyphrases": gt_keyphrases, 
            "hungarian similarity": similarity
            }

# get entire dataset
dataset = load_dataset("midas/krapivin", "raw")["test"]

test = dataset[0]
result = bench_sample("openrouter/google/gemini-2.0-flash-001", test)
print("generated keyphrases: " + result["generated keyphrases"])
print("ground truth keyphrases: " + result["ground truth keyphrases"])
print("hungarian similarity: " + result["hungarian similarity"])