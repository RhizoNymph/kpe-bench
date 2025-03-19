import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import wandb

def hungarian_similarity(predicted_embeddings, ground_truth_embeddings, penalty_cost=2.0):
    """
    Computes the optimal matching between predicted and ground truth embeddings.
    """
    similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
    cost_matrix = -similarity_matrix
    
    N, M = cost_matrix.shape
    size = max(N, M)
    padded_cost = np.full((size, size), penalty_cost, dtype=float)
    padded_cost[:N, :M] = cost_matrix
    
    row_ind, col_ind = linear_sum_assignment(padded_cost)
    
    valid_matches = [(i, j, similarity_matrix[i, j])
                     for i, j in zip(row_ind, col_ind) if i < N and j < M]
    matched_similarities = np.array([sim for (_, _, sim) in valid_matches])
    avg_similarity = np.mean(matched_similarities) if matched_similarities.size > 0 else 0.0
    
    return avg_similarity

def create_prompt(doc_text):
    return f"""Given the following text, extract the most relevant keywords or keyphrases that best summarize its content.
Try to minimize overlap of concepts in keyphrases.

Return the keywords as a comma-separated list of phrases.

Text: {doc_text}

Keywords:"""

def main():
    # Initialize wandb
    wandb.init(project="keyphrase-extraction-grpo")

    # Load models
    print("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Initialize model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure GRPO
    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        use_score_scaling=True,
        use_score_norm=True,
        max_steps=1000,
        logging_steps=10,
        output_dir="./keyphrase-extraction-grpo-checkpoints",
        save_steps=100,
        eval_steps=100,
        warmup_steps=100,
        log_with="wandb"
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("midas/krapivin", "raw")["test"]

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    generation_kwargs = {
        "min_length": 10,
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    print("Starting training...")
    best_reward = 0.0
    
    for epoch in range(3):  # Number of epochs
        epoch_rewards = []
        
        for batch_idx, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch}")):
            # Prepare input
            doc_text = " ".join(batch["document"])
            gt_keyphrases = batch["extractive_keyphrases"] + batch["abstractive_keyphrases"]
            prompt = create_prompt(doc_text)
            
            # Tokenize input
            encoded_input = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(model.device)
            
            # Generate responses
            outputs = trainer.generate(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                **generation_kwargs
            )
            
            # Process outputs and compute rewards
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_keyphrases = [kp.strip() for kp in response_text.split(",")]
            
            # Compute embeddings and reward
            generated_embeddings = [
                embedding_model.encode(kp, normalize_embeddings=True)
                for kp in generated_keyphrases
            ]
            gt_embeddings = [
                embedding_model.encode(kp, normalize_embeddings=True)
                for kp in gt_keyphrases
            ]
            
            reward = hungarian_similarity(
                np.array(generated_embeddings), 
                np.array(gt_embeddings)
            )
            epoch_rewards.append(reward)
            rewards = torch.tensor([reward], device=model.device)
            
            # GRPO step
            loss = trainer.compute_loss(
                model=model,
                inputs=encoded_input,
                outputs=outputs,
                rewards=rewards
            )
            
            # Backward pass and optimization
            loss.backward()
            
            if (batch_idx + 1) % grpo_config.gradient_accumulation_steps == 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            
            # Log metrics
            if batch_idx % grpo_config.logging_steps == 0:
                trainer.log({
                    "loss": loss.item(),
                    "reward": reward,
                    "epoch": epoch,
                    "batch": batch_idx,
                })
            
            # Save best model
            avg_epoch_reward = np.mean(epoch_rewards)
            if avg_epoch_reward > best_reward:
                best_reward = avg_epoch_reward
                model.save_pretrained(f"{grpo_config.output_dir}/best_model")
                tokenizer.save_pretrained(f"{grpo_config.output_dir}/best_model")
        
        # Log epoch metrics
        print(f"Epoch {epoch} average reward: {avg_epoch_reward:.4f}")
        wandb.log({
            "epoch": epoch,
            "epoch_reward": avg_epoch_reward,
        })

    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main()