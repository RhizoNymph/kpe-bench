# -*- coding: utf-8 -*-
"""Keyphrase Extraction Fine-tuning with GRPO"""

# All imports at the top
from unsloth import FastModel
import torch
from datasets import load_dataset, config
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer, Trainer, TrainingArguments
import wandb

# Set parameters
max_seq_length = 4096  # Increased significantly
max_prompt_length = 3840  # Most of the sequence length for input

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

# Load base model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
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

Document:
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
def match_format_exactly(prompts, completions, **kwargs):
    scores = []
    for prompt, completion in zip(prompts, completions):
        score = 0
        response = completion  # completion is already a string
        
        if len(scores) < 2:
            try:
                import shutil
                terminal_width = shutil.get_terminal_size().columns
            except:
                terminal_width = 80
            
            # Calculate width for each half (subtracting 3 for the separator)
            half_width = (terminal_width - 3) // 2
            divider = "-" * terminal_width
            
            # Split prompt and completion into lines that fit in half width
            def split_text(text, width):
                words = text[:1000].split()  # Limit to first 200 chars
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= width:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(" ".join(current_line))
                return lines
            
            prompt_lines = split_text(prompt, half_width)
            completion_lines = split_text(response, half_width)
            
            print(divider)
            print("PROMPT".ljust(half_width) + " | " + "COMPLETION".ljust(half_width))
            print("-" * half_width + "-+-" + "-" * half_width)
            
            # Print both sides line by line
            for i in range(max(len(prompt_lines), len(completion_lines))):
                left = prompt_lines[i] if i < len(prompt_lines) else ""
                right = completion_lines[i] if i < len(completion_lines) else ""
                print(f"{left:<{half_width}} | {right:<{half_width}}")
            
            print(divider + "\n")
        
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
    for completion in completions:
        score = 0
        response = completion  # completion is already a string
        
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
    scores = []
    
    if "ground_truth" not in kwargs:
        print("WARNING: ground_truth not found in kwargs")
        return [0.0] * len(completions)
    
    ground_truth_keyphrases = kwargs["ground_truth"]
    
    for completion, gt_keyphrases in zip(completions, ground_truth_keyphrases):
        response = completion  # completion is already a string
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
        
        # Get embeddings
        try:
            generated_embeddings = [
                embedding_model.encode(kp, normalize_embeddings=True) 
                for kp in generated_keyphrases
            ]
            gt_embeddings = [
                embedding_model.encode(kp, normalize_embeddings=True)
                for kp in gt_keyphrases
            ]
            
            # Use Hungarian algorithm for matching
            _, _, _, avg_similarity = hungarian_similarity(
                generated_embeddings, 
                gt_embeddings
            )

            print(f"Average similarity: {avg_similarity}")
            
            # Scale similarity to reasonable reward range
            reward = avg_similarity * 3.0  # Scale up to max ~3.0
            scores.append(reward)
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            scores.append(0)
            
    return scores

def prepare_document(doc):
    """Better document preparation with focus on important sections"""
    if isinstance(doc, list):
        doc = " ".join(doc)
    
    # Remove excessive whitespace and normalize
    doc = " ".join(doc.split())
    
    # If document is very long, try to keep most important parts
    words = doc.split()
    if len(words) > 3840:  # Using tokens would be more accurate, but words as approximation
        # Keep first 2000 words (usually abstract + intro) and last 1840 (usually conclusion/summary)
        doc = " ".join(words[:2000] + words[-1840:])
    
    return doc

# Add longer timeout and retry settings for dataset loading
config.HF_DATASETS_TIMEOUT = 100  # Increase timeout to 100 seconds
config.MAX_RETRIES = 3  # Add retries


dataset = load_dataset("midas/krapivin", "raw", split="test")

print("Dataset fields:", list(dataset[0].keys()))
print("Example document:", dataset[0]["document"][:2])
print("Example keyphrases:", dataset[0]["extractive_keyphrases"])

# Limit dataset size for initial training
max_samples = 100  # Adjust as needed
dataset = dataset.select(range(min(max_samples, len(dataset))))

# Map the dataset with better processing
dataset = dataset.map(lambda x: {
    "prompt": tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prepare_document(x["document"])}
    ], add_generation_prompt=True, tokenize=False),
    "ground_truth": list(set(  # Remove duplicates
        (x["extractive_keyphrases"] if isinstance(x["extractive_keyphrases"], list) else []) + 
        (x["abstractive_keyphrases"] if isinstance(x["abstractive_keyphrases"], list) else [])
    ))
})

# Add debug print to verify prompt format
print("\nVerifying prompt format:")
print(dataset[0]["prompt"][:500], "...\n")  # Print first 500 chars of first example

# Check if model understands format before training
print("Testing model's understanding of the format before training...")
test_doc = "Machine learning is a field of inquiry devoted to understanding and building methods that learn."
test_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_doc},
]
test_text = tokenizer.apply_chat_template(test_prompt, add_generation_prompt=True, tokenize=False)

# Create a custom streamer to capture output
class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt, test_doc):
        super().__init__(tokenizer, skip_prompt)
        self.generated_text = ""
        self.tokenizer = tokenizer
        self.started_generating = False
        self.test_doc = test_doc
        
    def split_text(self, text, width):
        words = text[:1000].split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        return lines
        
    def put(self, value):
        # Decode the tensor to text before adding to generated_text
        if torch.is_tensor(value):
            text = self.tokenizer.decode(value[0], skip_special_tokens=True)
        else:
            text = value
            
        # Only start capturing after we see "model" token
        if "model" in text:
            self.started_generating = True
            text = text.split("model")[-1]  # Only keep text after "model"
            
        if self.started_generating:
            self.generated_text += text
            # Also print the text as it's generated
            print(text, end="", flush=True)
        
    def end(self):
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80
        
        # Calculate width for each half
        half_width = (terminal_width - 3) // 2
        divider = "-" * terminal_width
        
        print("\n")  # Add some spacing before the side-by-side view
        
        # Get input and output lines using class method
        input_lines = self.split_text(self.test_doc, half_width)
        output_lines = self.split_text(self.generated_text.strip(), half_width)
        
        print(divider)
        print("INPUT".ljust(half_width) + " | " + "OUTPUT".ljust(half_width))
        print("-" * half_width + "-+-" + "-" * half_width)
        
        # Print both sides line by line
        for i in range(max(len(input_lines), len(output_lines))):
            left = input_lines[i] if i < len(input_lines) else ""
            right = output_lines[i] if i < len(output_lines) else ""
            print(f"{left:<{half_width}} | {right:<{half_width}}")
        
        print(divider + "\n")

streamer = CaptureStreamer(tokenizer, skip_prompt=True, test_doc=test_doc)

print("Initial generation:")
_ = model.generate(
    **tokenizer(test_text, return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    streamer=streamer,
)
print("Starting GRPO training...")

# Configure GRPO Trainer with wandb instead of tensorboard
training_args = GRPOConfig(
    learning_rate = 2e-6,  # Lower learning rate for semantic task
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,  # Higher beta2 for more stable training
    weight_decay = 0.01,
    warmup_ratio = 0.05,  # Longer warmup for semantic tasks
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    
    per_device_train_batch_size = 4,  # Larger batch size if memory allows
    gradient_accumulation_steps = 4,
    num_generations = 4,  # More generations for better reward estimation
    
    max_prompt_length = 3840,  # Much larger input context
    max_completion_length = 256,  # Keep this reasonable since outputs are keyphrases
    max_steps = 500,  # More steps for semantic learning
    save_steps = 100,
    max_grad_norm = 1.0,
    report_to = "wandb",  # Changed from "tensorboard" to "wandb"
    output_dir = "outputs-keyphrase",
)

# Ensure consistent tokenizer settings
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Add before trainer initialization
wandb.init(
    project="keyphrase-extraction",
    config={
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "model": "gemma-3b",
        "task": "keyphrase_extraction"
    }
)

# Update trainer initialization (remove formatting_func)
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        keyphrase_similarity_reward,
    ],
    reward_weights = [0.2, 0.1, 0.7],
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

def evaluate_model(model, tokenizer, eval_dataset, num_samples=10):
    """Evaluate model performance on a subset of data"""
    total_similarity = 0
    
    for i in range(num_samples):
        sample = eval_dataset[i]
        messages = sample["prompt"]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        outputs = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=64,
            temperature=0.7,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Compute similarity with ground truth...
        # Add similarity to total...
        
    return total_similarity / num_samples

# Add evaluation callback to training loop
