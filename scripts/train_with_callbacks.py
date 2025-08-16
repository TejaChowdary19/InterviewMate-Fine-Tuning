#!/usr/bin/env python3
"""
Comprehensive Training Script for InterviewMate Fine-tuning
Includes callbacks, logging, checkpointing, and validation
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import get_peft_model_state_dict

class InterviewMateTrainer:
    """Comprehensive trainer for InterviewMate fine-tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config['run_name']
        self.output_dir = Path(f"./results/{self.run_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"🚀 Initializing training for {self.run_name}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def setup_tokenizer(self):
        """Setup and configure tokenizer"""
        print("🔤 Setting up tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        print(f"✅ Tokenizer ready with vocabulary size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def setup_model(self):
        """Setup and configure model with LoRA"""
        print("🧠 Setting up model...")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize embeddings if needed
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"🔄 Resized embeddings to {len(self.tokenizer)}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=self.config['target_modules'],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"📊 Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")
        
        return self.model
    
    def setup_datasets(self):
        """Setup training and validation datasets"""
        print("📚 Setting up datasets...")
        
        # Load preprocessed datasets
        dataset_path = "data/tokenized_dataset"
        if not Path(dataset_path).exists():
            print("❌ Preprocessed dataset not found. Run data_preparation.py first!")
            return None, None
        
        datasets = load_from_disk(dataset_path)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors=None
            )
        
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names
        )
        
        print(f"✅ Datasets ready:")
        print(f"   Train: {len(tokenized_datasets['train'])} samples")
        print(f"   Validation: {len(tokenized_datasets['validation'])} samples")
        print(f"   Test: {len(tokenized_datasets['test'])} samples")
        
        return tokenized_datasets['train'], tokenized_datasets['validation']
    
    def setup_training_args(self):
        """Setup training arguments with comprehensive configuration"""
        print("⚙️ Setting up training arguments...")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            
            # Training parameters
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_steps=self.config['warmup_steps'],
            
            # Evaluation and logging
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=self.config['save_total_limit'],
            
            # Checkpointing
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Optimization
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            bf16=False,
            
            # Reporting
            report_to="none",
            logging_dir=str(self.output_dir / "logs"),
            
            # Reproducibility
            seed=self.config['seed'],
            dataloader_pin_memory=False,
            
            # Early stopping
            dataloader_num_workers=0,
        )
        
        return training_args
    
    def setup_data_collator(self):
        """Setup data collator for training"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
    def train(self):
        """Execute the complete training pipeline"""
        print("🎯 Starting training pipeline...")
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        train_dataset, eval_dataset = self.setup_datasets()
        
        if train_dataset is None:
            return
        
        training_args = self.setup_training_args()
        data_collator = self.setup_data_collator()
        
        # Setup trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("🚀 Starting training...")
        print(f"📊 Training config: {json.dumps(self.config, indent=2)}")
        
        # Train the model
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training results
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        print(f"✅ Training complete! Final model saved to {final_model_path}")
        print(f"📊 Training metrics: {train_result.metrics}")
        
        return trainer, train_result

def main():
    """Main training execution"""
    
    # Training configurations
    configs = [
        {
            "run_name": "run_A_improved",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42
        },
        {
            "run_name": "run_B_improved",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 4,
            "lora_alpha": 8,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42
        },
        {
            "run_name": "run_C_improved",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 2,
            "lora_alpha": 4,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42
        }
    ]
    
    # Run training for each configuration
    for config in configs:
        print(f"\n{'='*60}")
        print(f"🚀 Starting {config['run_name']}")
        print(f"{'='*60}")
        
        try:
            trainer = InterviewMateTrainer(config)
            trainer.train()
        except Exception as e:
            print(f"❌ Error in {config['run_name']}: {e}")
            continue
        
        print(f"✅ Completed {config['run_name']}\n")

if __name__ == "__main__":
    main()
