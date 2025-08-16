#!/usr/bin/env python3
"""
Optimized Training Script for InterviewMate
Enhanced performance, memory management, and advanced training features
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

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils import get_peft_model_state_dict
import accelerate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedInterviewMateTrainer:
    """Optimized trainer with advanced features and performance optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config['run_name']
        self.output_dir = Path(f"./results/{self.run_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance optimization flags
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_gradient_checkpointing = config.get('use_gradient_checkpointing', True)
        self.use_8bit_optimizer = config.get('use_8bit_optimizer', False)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Memory management
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        self.effective_batch_size = config['batch_size'] * self.gradient_accumulation_steps
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"🚀 Initializing optimized training for {self.run_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"⚡ Performance optimizations: Mixed Precision={self.use_mixed_precision}, "
                   f"Gradient Checkpointing={self.use_gradient_checkpointing}")
        
    @contextmanager
    def memory_tracker(self):
        """Context manager to track memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"💾 Initial GPU memory: {initial_memory:.2f} GB")
            
            try:
                yield
            finally:
                final_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logger.info(f"💾 Final GPU memory: {final_memory:.2f} GB")
                logger.info(f"💾 Peak GPU memory: {peak_memory:.2f} GB")
                torch.cuda.empty_cache()
        else:
            yield
    
    def setup_tokenizer(self):
        """Setup and configure tokenizer with optimizations"""
        logger.info("🔤 Setting up optimized tokenizer...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        # Optimize tokenizer settings
        self.tokenizer.padding_side = 'left'  # Better for causal LM
        self.tokenizer.truncation_side = 'right'
        
        logger.info(f"✅ Tokenizer ready with vocabulary size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def setup_model(self):
        """Setup and configure model with advanced optimizations"""
        logger.info("🧠 Setting up optimized model...")
        
        # Determine device and dtype
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.float16 if self.use_mixed_precision and torch.cuda.is_available() else torch.float32
        
        # Load base model with optimizations
        with self.memory_tracker():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # Resize embeddings if needed
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"🔄 Resized embeddings to {len(self.tokenizer)}")
        
        # Apply gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        
        # Configure LoRA with optimizations
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=self.config['target_modules'],
            lora_dropout=self.config['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            fan_in_fan_out=False,
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
        
        logger.info(f"📊 Trainable params: {trainable_params:,} || All params: {all_param:,} || "
                   f"Trainable%: {100 * trainable_params / all_param:.2f}%")
        
        return self.model
    
    def setup_datasets(self):
        """Setup training and validation datasets with optimizations"""
        logger.info("📚 Setting up optimized datasets...")
        
        # Load preprocessed datasets
        dataset_path = "data/tokenized_dataset"
        if not Path(dataset_path).exists():
            logger.error("❌ Preprocessed dataset not found. Run data_preparation.py first!")
            return None, None
        
        datasets = load_from_disk(dataset_path)
        
        # Optimized tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors=None,
                return_attention_mask=True
            )
        
        # Process datasets with optimizations
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
            num_proc=4,  # Parallel processing
            desc="Tokenizing datasets"
        )
        
        logger.info(f"✅ Datasets ready:")
        logger.info(f"   Train: {len(tokenized_datasets['train'])} samples")
        logger.info(f"   Validation: {len(tokenized_datasets['validation'])} samples")
        logger.info(f"   Test: {len(tokenized_datasets['test'])} samples")
        
        return tokenized_datasets['train'], tokenized_datasets['validation']
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimized optimizer and learning rate scheduler"""
        logger.info("⚙️ Setting up optimized optimizer and scheduler...")
        
        # Prepare optimizer
        if self.use_8bit_optimizer and torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                logger.info("✅ Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning("⚠️ 8-bit optimizer not available, using standard AdamW")
                optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay'],
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        
        # Calculate total steps
        total_steps = (len(self.train_dataset) // self.config['batch_size']) * self.config['num_epochs']
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"📈 Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        
        return optimizer, scheduler
    
    def setup_training_args(self):
        """Setup training arguments with performance optimizations"""
        logger.info("⚙️ Setting up optimized training arguments...")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            
            # Training parameters
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            warmup_steps=self.config['warmup_steps'],
            
            # Performance optimizations
            fp16=self.use_mixed_precision and torch.cuda.is_available(),
            bf16=False,  # Use FP16 for better compatibility
            dataloader_pin_memory=False,  # Disable for MPS compatibility
            dataloader_num_workers=0,  # Disable for stability
            
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
            
            # Advanced optimizations
            gradient_checkpointing=self.use_gradient_checkpointing,
            max_grad_norm=self.max_grad_norm,
            remove_unused_columns=False,
            
            # Reporting
            report_to="none",
            logging_dir=str(self.output_dir / "logs"),
            
            # Reproducibility
            seed=self.config['seed'],
            
            # Early stopping
        )
        
        return training_args
    
    def setup_data_collator(self):
        """Setup optimized data collator"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
    
    def create_custom_trainer(self, training_args, train_dataset, eval_dataset, 
                             data_collator, optimizer, scheduler):
        """Create custom trainer with optimizer and scheduler"""
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            optimizers=(optimizer, scheduler)
        )
        
        return trainer
    
    def train(self):
        """Execute the optimized training pipeline"""
        logger.info("🎯 Starting optimized training pipeline...")
        
        start_time = time.time()
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        self.train_dataset, self.eval_dataset = self.setup_datasets()
        
        if self.train_dataset is None:
            return
        
        training_args = self.setup_training_args()
        data_collator = self.setup_data_collator()
        optimizer, scheduler = self.setup_optimizer_and_scheduler()
        
        # Setup trainer
        trainer = self.create_custom_trainer(
            training_args, self.train_dataset, self.eval_dataset, 
            data_collator, optimizer, scheduler
        )
        
        logger.info("🚀 Starting optimized training...")
        logger.info(f"📊 Training config: {json.dumps(self.config, indent=2)}")
        
        # Train the model
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"⏱️ Total training time: {training_time/60:.2f} minutes")
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training results with additional metrics
        training_metrics = train_result.metrics
        training_metrics['total_training_time_minutes'] = training_time / 60
        training_metrics['effective_batch_size'] = self.effective_batch_size
        training_metrics['memory_optimizations'] = {
            'mixed_precision': self.use_mixed_precision,
            'gradient_checkpointing': self.use_gradient_checkpointing,
            '8bit_optimizer': self.use_8bit_optimizer
        }
        
        with open(self.output_dir / "training_results.json", 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        logger.info(f"✅ Training complete! Final model saved to {final_model_path}")
        logger.info(f"📊 Training metrics: {training_metrics}")
        
        # Clean up memory
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return trainer, train_result

def main():
    """Main optimized training execution"""
    
    # Enhanced training configurations with performance optimizations
    configs = [
        {
            "run_name": "run_A_optimized",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 2,  # Increased batch size
            "gradient_accumulation_steps": 2,  # Reduced for larger effective batch
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42,
            
            # Performance optimizations
            "use_mixed_precision": True,
            "use_gradient_checkpointing": True,
            "use_8bit_optimizer": False,  # Disable for stability
            "max_grad_norm": 1.0
        },
        {
            "run_name": "run_B_optimized",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 4,
            "lora_alpha": 8,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42,
            
            # Performance optimizations
            "use_mixed_precision": True,
            "use_gradient_checkpointing": True,
            "use_8bit_optimizer": False,
            "max_grad_norm": 1.0
        },
        {
            "run_name": "run_C_optimized",
            "model_name": "tiiuae/falcon-rw-1b",
            "lora_r": 2,
            "lora_alpha": 4,
            "target_modules": ["query_key_value"],
            "lora_dropout": 0.05,
            "max_length": 256,
            "num_epochs": 5,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 50,
            "eval_steps": 25,
            "logging_steps": 10,
            "save_steps": 25,
            "save_total_limit": 3,
            "seed": 42,
            
            # Performance optimizations
            "use_mixed_precision": True,
            "use_gradient_checkpointing": True,
            "use_8bit_optimizer": False,
            "max_grad_norm": 1.0
        }
    ]
    
    # Run training for each configuration
    for config in configs:
        print(f"\n{'='*60}")
        print(f"🚀 Starting {config['run_name']}")
        print(f"{'='*60}")
        
        try:
            trainer = OptimizedInterviewMateTrainer(config)
            trainer.train()
        except Exception as e:
            logger.error(f"❌ Error in {config['run_name']}: {e}")
            continue
        
        print(f"✅ Completed {config['run_name']}\n")

if __name__ == "__main__":
    main()
