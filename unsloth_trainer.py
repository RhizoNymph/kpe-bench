# -*- coding: utf-8 -*-
"""Keyphrase Extraction Fine-tuning with GRPO"""

# All imports at the top
from unsloth import FastModel
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer, Trainer, TrainingArguments

# Set parameters
max_seq_length = 1024
max_prompt_length = 512

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

# Load base model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)

# Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# Define keyphrase format tags
keyphrase_start = "<keyphrases>"
keyphrase_end = "</keyphrases>"

# Define system prompt
system_prompt = f"""You are given a document.
Extract the most relevant keywords or keyphrases that best summarize its content.
Try to minimize overlap of concepts in keyphrases.

IMPORTANT: You must return your answer in the following format:
{keyphrase_start}keyword1, keyword2, keyword3, ...{keyphrase_end}

Here are some examples:

Document: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
{keyphrase_start}machine learning, computer programming, automated learning{keyphrase_end}

Document: Climate change refers to long-term shifts in temperatures and weather patterns.
{keyphrase_start}climate change, global warming, weather patterns, long-term temperature shifts{keyphrase_end}

Now extract keyphrases from the document I will provide.
"""

# Define regex pattern for matching
match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{keyphrase_start}(.+?){keyphrase_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# Define Hungarian similarity function
def hungarian_similarity(predicted_embeddings, ground_truth_embeddings, penalty_cost=2.0):
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

# Define reward functions
def match_format_exactly(completions, **kwargs):
    scores = []
    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        
        # Debug output for first few completions
        if i < 2:
            print(f"Completion {i}: {response[:100]}...")
        
        # Give partial credit for just having the tags
        if keyphrase_start in response and keyphrase_end in response:
            # Get the text between the tags (even if not a perfect match)
            start_idx = response.find(keyphrase_start) + len(keyphrase_start)
            end_idx = response.find(keyphrase_end, start_idx)
            
            if start_idx > 0 and end_idx > start_idx:
                extracted = response[start_idx:end_idx].strip()
                if extracted and "," in extracted:
                    # Good response with multiple keyphrases
                    score += 2.0
                elif extracted:
                    # At least something between tags
                    score += 1.0
            
        # Also check for exact format match
        if match_format.search(response) is not None:
            score += 1.0  # Bonus for exact format match
            
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        
        # Check for individual tags with partial scoring
        if keyphrase_start in response:
            score += 0.5
        else:
            score -= 0.3
            
        if keyphrase_end in response:
            score += 0.5
        else:
            score -= 0.3
            
        scores.append(score)
    return scores

def keyphrase_similarity_reward(prompts, completions, **kwargs):
    # Simple debugging
    print(f"Available kwargs: {list(kwargs.keys())}")
    if completions:
        print(f"Sample completion: {completions[0][0]['content'][:100]}...")
    
    scores = []
    
    # Check if ground_truth is available in kwargs
    if "ground_truth" not in kwargs:
        print("WARNING: ground_truth not found in kwargs")
        # Fall back to basic format matching
        for completion in completions:
            response = completion[0]["content"]
            match = match_format.search(response)
            scores.append(1.0 if match else 0.0)
        return scores
    
    ground_truth_keyphrases = kwargs["ground_truth"]
    
    # Simple check that vectors are properly aligned
    if len(completions) != len(ground_truth_keyphrases):
        print(f"WARNING: Mismatch in lengths - completions:{len(completions)}, ground_truth:{len(ground_truth_keyphrases)}")
        # Add zeros for any missing items
        max_len = max(len(completions), len(ground_truth_keyphrases))
        return [0.0] * max_len
    
    # Process each completion
    for i, (completion, gt_keyphrases) in enumerate(zip(completions, ground_truth_keyphrases)):
        response = completion[0]["content"]
        match = match_format.search(response)
        
        if match is None:
            scores.append(0)
            continue
            
        # Extract keyphrases from the response
        extracted_text = match.group(1).strip()
        generated_keyphrases = [kp.strip() for kp in extracted_text.split(",") if kp.strip()]
        
        if not generated_keyphrases:
            scores.append(0)
            continue
        
        # For initial training, just reward for generating any keyphrases
        # This gets training started without embedding complexity
        reward = min(len(generated_keyphrases) * 0.5, 3.0)  # Cap at 3
        scores.append(reward)
        
        # Print for debugging
        if i < 2:
            print(f"Generated: {generated_keyphrases}")
            print(f"Ground truth: {gt_keyphrases}")
            print(f"Basic reward: {reward}")
    
    return scores

# Add a max document length to avoid memory issues with very long documents
def truncate_document(doc, max_length=512):
    if isinstance(doc, list):
        joined = " ".join(doc)
        words = joined.split()
        return " ".join(words[:max_length])
    else:
        words = str(doc).split()
        return " ".join(words[:max_length])

# Load and prepare dataset
dataset = load_dataset("midas/krapivin", "raw", split = "test")
print("Dataset fields:", list(dataset[0].keys()))
print("Example document:", dataset[0]["document"][:2])
print("Example keyphrases:", dataset[0]["extractive_keyphrases"])

# Limit dataset size for initial training
max_samples = 100  # Adjust as needed
dataset = dataset.select(range(min(max_samples, len(dataset))))

# Map the dataset
dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": truncate_document(x["document"])},
    ],
    "ground_truth": (x["extractive_keyphrases"] if isinstance(x["extractive_keyphrases"], list) else []) + 
                   (x["abstractive_keyphrases"] if isinstance(x["abstractive_keyphrases"], list) else [])
})

# Print example after mapping to verify
print("Mapped dataset example:")
print("Prompt:", dataset[0]["prompt"])
print("Ground truth keyphrases:", dataset[0]["ground_truth"])
print("\nChecking dataset structure:")
print("First example keys:", dataset[0].keys())
print("First example ground truth type:", type(dataset[0]["ground_truth"]))
print("First example ground truth sample:", dataset[0]["ground_truth"][:3] if dataset[0]["ground_truth"] else "Empty")

# Check if model understands format before training
print("Testing model's understanding of the format before training...")
test_doc = "Machine learning is a field of inquiry devoted to understanding and building methods that learn."
test_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_doc},
]
test_text = tokenizer.apply_chat_template(test_prompt, add_generation_prompt=True, tokenize=False)
print("Initial generation:")
_ = model.generate(
    **tokenizer(test_text, return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print("Starting GRPO training...")

# Configure GRPO Trainer with values closer to what works for GSM8K
training_args = GRPOConfig(
    learning_rate = 5e-6,  # Slightly higher learning rate - math reasoning usually needs this
    adam_beta1 = 0.9,
    adam_beta2 = 0.95,  # Slightly lower beta2 for more adaptation
    weight_decay = 0.05,  # More regularization
    warmup_ratio = 0.03,  # Shorter warmup
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    num_generations = 2,
    
    max_prompt_length = 256,
    max_completion_length = 128,  # Slightly longer completion length
    max_steps = 100,
    save_steps = 25,
    max_grad_norm = 1.0,
    report_to = "none",
    output_dir = "outputs-keyphrase",
)

# Ensure consistent tokenizer settings
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Create and run the trainer with weighted reward functions
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        keyphrase_similarity_reward,
    ],
    reward_weights = [0.3, 0.2, 0.5],  # Weight similarity higher
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# Inference example
sample_doc = "Machine learning is a field of inquiry devoted to understanding and building methods that learn, that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": sample_doc},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 64,
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# Saving the model
model.save_pretrained("gemma-3-keyphrase")
tokenizer.save_pretrained("gemma-3-keyphrase")
