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
    """
    Compute the optimal matching between predicted and ground truth embeddings 
    using the Hungarian algorithm and return the indices, matched similarities,
    and average similarity.
    """
    similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
    
    cost_matrix = -similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_similarities = similarity_matrix[row_ind, col_ind]
    
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
print(f"Hungarian matched indices (generated, ground truth): {list(zip(rows, cols))}")
print(f"Matched similarities: {matched_sim}")
print(f"Average similarity: {avg_sim}")
