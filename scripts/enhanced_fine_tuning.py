#!/usr/bin/env python3
"""
Enhanced Fine-tuning Script for InterviewMate
Uses expanded dataset (905 examples) with advanced techniques
"""

import os
import json
import torch
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging
from contextlib import contextmanager

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import get_peft_model_state_dict
import accelerate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInterviewMateTrainer:
    def __init__(self, 
                 base_model_name: str = "tiiuae/falcon-rw-1b",
                 use_mixed_precision: bool = True,
                 use_gradient_checkpointing: bool = True,
                 max_grad_norm: float = 1.0):
        
        self.base_model_name = base_model_name
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Memory tracking
        self.memory_tracker = MemoryTracker()
    
    @contextmanager
    def memory_tracker(self):
        """Context manager for memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        yield
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def setup_tokenizer(self):
        """Setup tokenizer with enhanced configuration"""
        logger.info("Setting up tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add special tokens for instruction tuning
        special_tokens = {
            "pad_token": "<PAD>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>"
        }
        
        # Add instruction-specific tokens
        instruction_tokens = {
            "additional_special_tokens": [
                "<QUESTION>", "</QUESTION>",
                "<ANSWER>", "</ANSWER>",
                "<INSTRUCTION>", "</INSTRUCTION>"
            ]
        }
        
        # Update tokenizer
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.add_special_tokens(instruction_tokens)
        
        logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def setup_model(self):
        """Setup model with enhanced LoRA configuration"""
        logger.info("Setting up model...")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.use_mixed_precision else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enhanced LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,  # Increased rank for better performance
            lora_alpha=64,  # Increased alpha for better scaling
            lora_dropout=0.1,  # Added dropout for regularization
            target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            bias="none",
            use_rslora=True,  # Use RSLoRA for better stability
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def setup_datasets(self):
        """Setup enhanced datasets with instruction tuning format"""
        logger.info("Setting up enhanced datasets...")
        
        # Load enhanced dataset
        with open("data/enhanced_ai_engineer_dataset.json", "r") as f:
            enhanced_data = json.load(f)
        
        # Convert to instruction tuning format
        instruction_data = []
        for item in enhanced_data:
            # Create instruction format
            instruction = f"""<INSTRUCTION>
You are an expert AI interview coach specializing in machine learning and data science. 
Provide comprehensive, accurate, and helpful answers to interview questions.
</INSTRUCTION>

<QUESTION>
{item['question']}
</QUESTION>

<ANSWER>
{item['answer']}
</ANSWER>"""
            
            instruction_data.append({"text": instruction})
        
        # Create dataset
        dataset = Dataset.from_list(instruction_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,  # Increased for longer answers
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Evaluation examples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup enhanced optimizer and scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Enhanced optimizer with better parameters
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # Lower learning rate for stability
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=True
        )
        
        # Enhanced scheduler with warmup and cosine annealing
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
            num_training_steps=num_training_steps
        )
        
        return optimizer, scheduler
    
    def setup_training_args(self, num_training_steps: int):
        """Setup enhanced training arguments"""
        logger.info("Setting up training arguments...")
        
        training_args = TrainingArguments(
            output_dir="./results/enhanced_training",
            num_train_epochs=5,  # Increased epochs for better learning
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,  # Effective batch size: 16
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=int(0.1 * num_training_steps),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.use_mixed_precision and torch.cuda.is_available(),
            bf16=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            gradient_checkpointing=self.use_gradient_checkpointing,
            max_grad_norm=self.max_grad_norm,
            report_to="tensorboard",
            run_name=f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            # Enhanced logging
            logging_dir="./results/enhanced_training/logs",
            log_level="info",
            # Enhanced evaluation
            eval_accumulation_steps=4,
            # Enhanced saving
            save_safetensors=True,
            # Enhanced monitoring
            dataloader_drop_last=True,
            # Enhanced optimization
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            # Enhanced regularization
            warmup_ratio=0.1
        )
        
        return training_args
    
    def setup_data_collator(self):
        """Setup enhanced data collator"""
        logger.info("Setting up data collator...")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        return data_collator
    
    def create_custom_trainer(self, train_dataset, eval_dataset, training_args, data_collator):
        """Create enhanced trainer with custom callbacks"""
        logger.info("Creating enhanced trainer...")
        
        # Enhanced callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ]
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=self.compute_metrics
        )
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute enhanced evaluation metrics"""
        predictions, labels = eval_pred
        
        # Convert predictions to text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple metrics for now (can be enhanced with ROUGE, BLEU, etc.)
        metrics = {
            "eval_loss": eval_pred.loss,
            "eval_samples": len(decoded_preds)
        }
        
        return metrics
    
    def train(self):
        """Execute enhanced training"""
        logger.info("Starting enhanced training...")
        
        start_time = time.time()
        
        try:
            # Setup components
            self.setup_tokenizer()
            self.setup_model()
            train_dataset, eval_dataset = self.setup_datasets()
            
            # Calculate training steps
            num_training_steps = len(train_dataset) // 2 * 5  # batch_size * epochs
            
            # Setup training components
            training_args = self.setup_training_args(num_training_steps)
            data_collator = self.setup_data_collator()
            
            # Create trainer
            trainer = self.create_custom_trainer(
                train_dataset, eval_dataset, training_args, data_collator
            )
            
            # Train model
            logger.info("Training model...")
            trainer.train()
            
            # Save final model
            logger.info("Saving enhanced model...")
            trainer.save_model("./results/enhanced_model")
            
            # Save tokenizer
            self.tokenizer.save_pretrained("./results/enhanced_model")
            
            # Evaluate final model
            logger.info("Evaluating enhanced model...")
            eval_results = trainer.evaluate()
            
            training_time = time.time() - start_time
            
            # Save training results
            results = {
                "training_time_minutes": training_time / 60,
                "final_eval_loss": eval_results["eval_loss"],
                "total_steps": trainer.state.global_step,
                "model_path": "./results/enhanced_model",
                "dataset_size": len(train_dataset) + len(eval_dataset),
                "enhancement": "Advanced instruction tuning with 905 examples"
            }
            
            with open("./results/enhanced_training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info("Enhanced training completed successfully!")
            logger.info(f"Training time: {training_time/60:.2f} minutes")
            logger.info(f"Final eval loss: {eval_results['eval_loss']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise e

class MemoryTracker:
    """Memory tracking utility"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
    
    def get_memory_info(self):
        """Get current memory information"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "reserved": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
            }
        elif torch.backends.mps.is_available():
            return {"device": "MPS", "info": "Memory tracking not available for MPS"}
        else:
            return {"device": "CPU", "info": "Memory tracking not available for CPU"}

def main():
    """Main function for enhanced fine-tuning"""
    logger.info("🚀 Starting Enhanced InterviewMate Fine-tuning...")
    
    # Create enhanced trainer
    trainer = EnhancedInterviewMateTrainer(
        base_model_name="tiiuae/falcon-rw-1b",
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        max_grad_norm=1.0
    )
    
    # Execute training
    results = trainer.train()
    
    logger.info("🎉 Enhanced fine-tuning completed!")
    logger.info(f"Results: {results}")

if __name__ == "__main__":
    main()
