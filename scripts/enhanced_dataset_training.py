#!/usr/bin/env python3
"""
Enhanced Dataset Training Script for InterviewMate
Uses the working approach but with the expanded 905-example dataset
"""

import json
import torch
import time
from pathlib import Path
from datetime import datetime
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_dataset():
    """Load the enhanced dataset"""
    logger.info("Loading enhanced dataset...")
    
    with open("data/enhanced_ai_engineer_dataset.json", "r") as f:
        enhanced_data = json.load(f)
    
    # Convert to training format (same as original)
    training_data = []
    for item in enhanced_data:
        training_data.append({
            "text": f"Question: {item['question']}\nAnswer: {item['answer']}"
        })
    
    logger.info(f"Loaded {len(training_data)} training examples")
    return training_data

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset"""
    logger.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Convert to HuggingFace dataset format
    from datasets import Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=hf_dataset.column_names
    )
    
    logger.info(f"Tokenized {len(tokenized_dataset)} examples")
    return tokenized_dataset

def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    logger.info("Setting up model and tokenizer...")
    
    # Load base model and tokenizer
    model_name = "tiiuae/falcon-rw-1b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Apply LoRA (same configuration as working script)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def setup_training_args():
    """Setup training arguments"""
    logger.info("Setting up training arguments...")
    
    training_args = TrainingArguments(
        output_dir="./results/enhanced_dataset_training",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="tensorboard",
        run_name=f"enhanced_dataset_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        logging_dir="./results/enhanced_dataset_training/logs",
        log_level="info",
        save_safetensors=True,
        dataloader_drop_last=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    return training_args

def main():
    """Main training function"""
    logger.info("🚀 Starting Enhanced Dataset Training...")
    logger.info("Using the expanded 905-example dataset!")
    
    start_time = time.time()
    
    try:
        # Setup components
        model, tokenizer = setup_model_and_tokenizer()
        
        # Load and tokenize dataset
        dataset = load_enhanced_dataset()
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Setup training arguments
        training_args = setup_training_args()
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Train model
        logger.info("Starting training on enhanced dataset...")
        trainer.train()
        
        # Save final model
        logger.info("Saving enhanced model...")
        trainer.save_model("./results/enhanced_dataset_model")
        tokenizer.save_pretrained("./results/enhanced_dataset_model")
        
        training_time = time.time() - start_time
        
        # Save training results
        results = {
            "training_time_minutes": training_time / 60,
            "model_path": "./results/enhanced_dataset_model",
            "dataset_size": len(dataset),
            "enhancement": "Training on expanded 905-example dataset",
            "improvement": "200% more training data than original"
        }
        
        with open("./results/enhanced_dataset_training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Enhanced dataset training completed successfully!")
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        logger.info(f"Model saved to: ./results/enhanced_dataset_model")
        logger.info(f"Trained on {len(dataset)} examples (vs 302 original)")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

