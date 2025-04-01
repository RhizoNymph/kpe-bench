from unsloth import FastModel, FastLanguageModel
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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from collections import Counter

# Set parameters
max_seq_length = 4096  # Increased significantly
max_prompt_length = 3840  # Most of the sequence length for input

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name="/home/nymph/.cache/huggingface/hub/models--unsloth--Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
    model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

# Define keyphrase format tags
keyphrase_start = "<keyphrases>"
keyphrase_end = "</keyphrases>"

# Define system prompt
system_prompt = """You are a keyphrase extraction assistant. Your only task is to extract relevant keyphrases from documents.

IMPORTANT: You must ONLY return keyphrases in this exact format:
<keyphrases>keyphrase1, keyphrase2, keyphrase3, ...</keyphrases>

Remember:
- Only output the keyphrases in the specified format
- Do not include any other text or explanation
- Separate keyphrases with commas"""

def prepare_document(doc):
    """Better document preparation with focus on important sections"""
    # Remove excessive whitespace and normalize
    doc = " ".join(doc)
    return doc

# Load dataset and limit if needed
dataset = load_dataset("midas/kp20k", "raw", split="train")
print("Dataset fields:", list(dataset[0].keys()))
print("Example document:", dataset[0]["document"])
print("Example keyphrases:", dataset[0]["extractive_keyphrases"])
print("Dataset length:", len(dataset))

max_samples = None
if max_samples and max_samples < len(dataset):
    print(f"Limiting dataset to {max_samples} samples")
    dataset = dataset.select(range(min(max_samples, len(dataset))))

# Calculate keyphrase statistics first
print("\nCalculating keyphrase statistics...")
keyphrase_counts = []
for sample in tqdm(dataset, desc="Counting keyphrases"):
    combined_keyphrases = list(set(
        (sample["extractive_keyphrases"] if isinstance(sample["extractive_keyphrases"], list) else []) + 
        (sample["abstractive_keyphrases"] if isinstance(sample["abstractive_keyphrases"], list) else [])
    ))
    if len(combined_keyphrases) > 0:  # Only include documents with at least one keyphrase
        keyphrase_counts.append(len(combined_keyphrases))

print(f"\nTotal documents: {len(dataset):,}")
print(f"Documents with keyphrases: {len(keyphrase_counts):,}")
print(f"Documents with no keyphrases: {len(dataset) - len(keyphrase_counts):,}")

min_keyphrases = min(keyphrase_counts)
max_keyphrases = max(keyphrase_counts)
avg_keyphrases = sum(keyphrase_counts) / len(keyphrase_counts)
median_keyphrases = sorted(keyphrase_counts)[len(keyphrase_counts)//2]

print(f"\nKeyphrase Statistics (excluding documents with no keyphrases):")
print(f"Minimum keyphrases per document: {min_keyphrases:,}")
print(f"Maximum keyphrases per document: {max_keyphrases:,}")
print(f"Average keyphrases per document: {avg_keyphrases:.1f}")
print(f"Median keyphrases per document: {median_keyphrases:,}")
print(f"Total keyphrases in dataset: {sum(keyphrase_counts):,}\n")

def count_tokens_for_sample(doc_text):    
    """Helper function to count tokens for a single sample"""
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": doc_text}
    ], add_generation_prompt=True, tokenize=False)
    tokens = len(tokenizer(prompt)["input_ids"][0])
    return tokens

# Calculate dataset token statistics using ThreadPoolExecutor
print("\nCalculating dataset token statistics...")
doc_token_counts = [None] * len(dataset)
num_workers = max(1, cpu_count() - 2)
print(f"Using {num_workers} workers for parallel processing...")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {}
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Submitting tasks"):
        doc_text = prepare_document(sample["document"])
        futures[executor.submit(count_tokens_for_sample, doc_text)] = i
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing results"):
        try:
            token_count = future.result()
            original_index = futures[future]
            doc_token_counts[original_index] = token_count
        except Exception as e:
            print(f"Error processing sample: {e}")
            doc_token_counts[futures[future]] = 0

doc_token_counts = [count for count in doc_token_counts if count is not None]
min_tokens = min(doc_token_counts)
max_tokens = max(doc_token_counts)
avg_tokens = sum(doc_token_counts) / len(doc_token_counts)
median_tokens = sorted(doc_token_counts)[len(doc_token_counts)//2]

print(f"\nDataset Token Statistics:")
print(f"Minimum tokens per document: {min_tokens:,}")
print(f"Maximum tokens per document: {max_tokens:,}")
print(f"Average tokens per document: {avg_tokens:,.1f}")
print(f"Median tokens per document: {median_tokens:,}")
print(f"Total tokens in dataset: {sum(doc_token_counts):,}\n")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# Precompile regex pattern for matching keyphrase format
match_format = re.compile(
    rf"^[\s]*{keyphrase_start}(.+?){keyphrase_end}[\s]*$",
    flags=re.MULTILINE | re.DOTALL
)

###############################################################################
# Improved Reward Functions
###############################################################################

def improved_format_reward(prompts, completions, **kwargs):
    """
    Computes a continuous reward based on how well the completion follows the
    desired keyphrase format. Rewards are given for:
      - Presence of start and end tags.
      - A regex match to extract keyphrases.
      - A number of keyphrases near the desired range (ideally 3 to 6).
    """
    scores = []
    for prompt, completion in zip(prompts, completions):
        response = completion.strip()
        score = 0.0

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
        
        # Check for the required start/end tags
        if keyphrase_start in response:
            score += 0.3
        else:
            score -= 0.3
        if keyphrase_end in response:
            score += 0.3
        else:
            score -= 0.3

        # Try to extract keyphrases using the regex pattern
        match = re.search(match_format, response)
        if match:
            content = match.group(1).strip()
            keyphrases = [k.strip() for k in content.split(",") if k.strip()]
            # Provide partial credit: ideal if between 3 and 6 keyphrases
            if 3 <= len(keyphrases) <= 6:
                score += 0.4
            else:
                # If off by one, give a bit of credit; otherwise penalize slightly
                diff = min(abs(len(keyphrases) - 3), abs(len(keyphrases) - 6))
                if diff == 1:
                    score += 0.2
                else:
                    score -= 0.2
        else:
            score -= 0.4

        # Clip score to the range [-1, 1]
        score = max(-1.0, min(1.0, score))
        scores.append(score)
    print(f"Improved format scores: {scores}")
    return scores

def improved_semantic_reward(prompts, completions, **kwargs):
    """
    Computes semantic reward using many-to-many matching approach:
    1. For each predicted keyphrase, finds its best matching ground truth phrases
    2. For each ground truth keyphrase, finds its best matching predictions
    3. Balances precision and recall-like metrics
    4. Penalizes redundancy within predictions
    """
    ground_truth = kwargs["ground_truth"]
    scores = []
    
    for completion in completions:
        response = completion.strip()
        reward = -1.0  # Default penalty
        match_pred = re.search(match_format, response)
        match_gt = re.search(match_format, ground_truth[0])
        
        if match_pred and match_gt:
            pred_content = match_pred.group(1).strip()
            gt_content = match_gt.group(1).strip()
            pred_keyphrases = [k.strip() for k in pred_content.split(",") if k.strip()]
            gt_keyphrases = [k.strip() for k in gt_content.split(",") if k.strip()]
            
            if pred_keyphrases and gt_keyphrases:
                # Get embeddings
                pred_embeddings = embedding_model.encode(pred_keyphrases)
                gt_embeddings = embedding_model.encode(gt_keyphrases)
                
                # Compute similarity matrix
                similarity_matrix = cosine_similarity(pred_embeddings, gt_embeddings)
                
                # Precision-like score: how well each prediction matches ground truth
                precision_score = similarity_matrix.max(axis=1).mean()
                
                # Recall-like score: how well ground truth is covered by predictions
                recall_score = similarity_matrix.max(axis=0).mean()
                
                # Check for redundancy within predictions
                pred_self_sim = cosine_similarity(pred_embeddings)
                np.fill_diagonal(pred_self_sim, 0)
                redundancy_penalty = 0
                if len(pred_self_sim) > 1:
                    # Calculate average similarity between different predictions
                    redundancy_penalty = pred_self_sim.mean()
                
                # Combine scores with F1-like balancing
                semantic_score = 2 * (precision_score * recall_score) / (precision_score + recall_score + 1e-8)
                
                # Apply redundancy penalty
                reward = semantic_score * (1 - 0.5 * redundancy_penalty)
                
                # Scale to [-1, 1]
                reward = 2 * reward - 1
                
        scores.append(reward)
    print(f"Improved semantic scores: {scores}")
    return scores

###############################################################################
# Dataset Mapping & Prompt Preparation
###############################################################################

# Update dataset mapping to wrap keyphrases in tags
dataset = dataset.map(lambda x: {
    "prompt": tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prepare_document(x["document"])}
    ], add_generation_prompt=True, tokenize=False),
    "ground_truth": f"{keyphrase_start}" + ", ".join(list(set(
        (x["extractive_keyphrases"] if isinstance(x["extractive_keyphrases"], list) else []) + 
        (x["abstractive_keyphrases"] if isinstance(x["abstractive_keyphrases"], list) else [])
    ))) + f"{keyphrase_end}"
})

print("\nVerifying prompt format:")
print(dataset[0]["prompt"][:500], "...\n")

###############################################################################
# Testing Model & Streamer Setup (Unchanged)
###############################################################################

print("Testing model's understanding of the format before training...")
test_doc = "Machine learning is a field of inquiry devoted to understanding and building methods that learn."
test_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_doc},
]
test_text = tokenizer.apply_chat_template(test_prompt, add_generation_prompt=True, tokenize=False)

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
        if torch.is_tensor(value):
            text = self.tokenizer.decode(value[0], skip_special_tokens=True)
        else:
            text = value
        if "model" in text:
            self.started_generating = True
            text = text.split("model")[-1]
        if self.started_generating:
            self.generated_text += text
            print(text, end="", flush=True)
        
    def end(self):
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80
        half_width = (terminal_width - 3) // 2
        divider = "-" * terminal_width
        print("\n")
        input_lines = self.split_text(self.test_doc, half_width)
        output_lines = self.split_text(self.generated_text.strip(), half_width)
        print(divider)
        print("INPUT".ljust(half_width) + " | " + "OUTPUT".ljust(half_width))
        print("-" * half_width + "-+-" + "-" * half_width)
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
print("Note: You might have to wait 150-200 steps to see improvement.")
print("The model may get 0 reward for the first 100 steps. Please be patient!")

###############################################################################
# GRPO Trainer Setup with Improved Reward Functions
###############################################################################

training_args = GRPOConfig(
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.05,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,
    temperature=1.5,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=500,
    save_steps=50,
    max_grad_norm=1.0,
    report_to="wandb",
    # scale_rewards=False,
    # num_iterations=4
    # use_vllm=True
)

tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

wandb.init(
    project="keyphrase-extraction",
    config={
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "model": "gemma-3-12b-it-unsloth-bnb-4bit",
        "task": "kp20k-keyphrase-extraction"
    }
)

# Use the improved reward functions here.
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[improved_format_reward, improved_semantic_reward],
    reward_weights=[0.3, 0.7],
    args=training_args,
    train_dataset=dataset
)

# Quick test of the new reward functions
print("\nTesting improved reward functions...")
test_prompts = ["test prompt"]
test_completions = [
    "<keyphrases>good keyphrase, another good one</keyphrases>",
    "bad format no tags",
    "<keyphrases>duplicate, duplicate</keyphrases>"
]
test_kwargs = {"ground_truth": ["<keyphrases>ground truth, test phrase</keyphrases>"]}

print("\nTesting improved format reward:")
improved_format_scores = improved_format_reward(test_prompts * 3, test_completions, **test_kwargs)
print(f"Improved Format scores: {improved_format_scores}")

print("\nTesting improved semantic reward:")
improved_semantic_scores = improved_semantic_reward(test_prompts * 3, test_completions, **test_kwargs)
print(f"Improved Semantic scores: {improved_semantic_scores}")

trainer.train()

# Inference example
sample_doc = ("Machine learning is a field of inquiry devoted to understanding and building methods that learn, "
              "that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part "
              "of artificial intelligence.")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": sample_doc},
]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=1.0, top_p=0.95, top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)

# Saving the model
model.save_pretrained("gemma-3-keyphrase")
tokenizer.save_pretrained("gemma-3-keyphrase")