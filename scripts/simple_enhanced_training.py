#!/usr/bin/env python3
"""
Simple Enhanced Training Script for InterviewMate
Compatible version that works with the expanded dataset
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
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
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
    
    # Convert to training format
    training_data = []
    for item in enhanced_data:
        training_data.append({
            "text": f"Question: {item['question']}\nAnswer: {item['answer']}"
        })
    
    logger.info(f"Loaded {len(training_data)} training examples")
    return training_data

def tokenize_dataset(dataset, tokenizer, max_length=1024):
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
    
    # Split dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

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
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Use standard rank for compatibility
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
    
    # Check if we're using MPS (Apple Silicon)
    use_fp16 = torch.cuda.is_available() and not torch.backends.mps.is_available()
    
    training_args = TrainingArguments(
            output_dir="./results/enhanced_training_simple",
            num_train_epochs=3,  # Start with 3 epochs
            per_device_train_batch_size=1,  # Reduce batch size for compatibility
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Effective batch size: 16
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,  # Only enable fp16 for CUDA
            remove_unused_columns=False,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            report_to="tensorboard",
            run_name=f"enhanced_training_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            logging_dir="./results/enhanced_training_simple/logs",
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
    logger.info("🚀 Starting Simple Enhanced InterviewMate Training...")
    
    start_time = time.time()
    
    try:
        # Setup components
        model, tokenizer = setup_model_and_tokenizer()
        
        # Load and tokenize dataset
        dataset = load_enhanced_dataset()
        train_dataset, eval_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Setup training arguments
        training_args = setup_training_args()
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001
            )
        ]
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving enhanced model...")
        trainer.save_model("./results/enhanced_model_simple")
        tokenizer.save_pretrained("./results/enhanced_model_simple")
        
        # Evaluate final model
        logger.info("Evaluating enhanced model...")
        eval_results = trainer.evaluate()
        
        training_time = time.time() - start_time
        
        # Save training results
        results = {
            "training_time_minutes": training_time / 60,
            "final_eval_loss": eval_results["eval_loss"],
            "total_steps": trainer.state.global_step,
            "model_path": "./results/enhanced_model_simple",
            "dataset_size": len(train_dataset) + len(eval_dataset),
            "enhancement": "Simple enhanced training with 905 examples"
        }
        
        with open("./results/enhanced_training_simple_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Enhanced training completed successfully!")
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        logger.info(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        logger.info(f"Model saved to: ./results/enhanced_model_simple")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
