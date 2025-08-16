#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for InterviewMate
Evaluates baseline vs fine-tuned models with multiple metrics
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

class InterviewMateEvaluator:
    """Comprehensive evaluator for InterviewMate models"""
    
    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        
    def load_baseline_model(self):
        """Load the baseline (unfine-tuned) model"""
        print("🧠 Loading baseline model...")
        
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.add_special_tokens({'pad_token': self.baseline_tokenizer.eos_token})
        
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.baseline_model.eval()
        
        print("✅ Baseline model loaded")
        
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        print(f"🎯 Loading fine-tuned model from {model_path}...")
        
        # Load tokenizer
        if Path(model_path).exists():
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.add_special_tokens({'pad_token': self.finetuned_tokenizer.eos_token})
        
        # Load base model
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load LoRA weights
        self.finetuned_model = PeftModel.from_pretrained(self.finetuned_model, model_path)
        self.finetuned_model.eval()
        
        print("✅ Fine-tuned model loaded")
        
    def load_test_data(self):
        """Load test dataset"""
        print("📚 Loading test data...")
        
        # Try to load preprocessed dataset first
        dataset_path = "data/tokenized_dataset"
        if Path(dataset_path).exists():
            datasets = load_from_disk(dataset_path)
            self.test_data = datasets['test']
        else:
            # Fallback to original dataset
            with open("data/ai_engineer_dataset.json", 'r') as f:
                data = json.load(f)
            # Use last 15% as test set
            test_size = int(len(data) * 0.15)
            self.test_data = data[-test_size:]
        
        print(f"✅ Test data loaded: {len(self.test_data)} samples")
        
    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 100) -> str:
        """Generate response from a model"""
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def calculate_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate multiple evaluation metrics"""
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        metrics.update({
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        })
        
        # BLEU score
        try:
            bleu_score = self.bleu.compute(
                predictions=predictions,
                references=references
            )
            metrics['bleu'] = bleu_score['bleu']
        except:
            metrics['bleu'] = 0.0
        
        # METEOR score
        try:
            meteor_score = self.meteor.compute(
                predictions=predictions,
                references=references
            )
            metrics['meteor'] = meteor_score['meteor']
        except:
            metrics['meteor'] = 0.0
        
        # Exact match
        exact_matches = sum(1 for ref, pred in zip(references, predictions) 
                          if ref.strip().lower() == pred.strip().lower())
        metrics['exact_match'] = exact_matches / len(references)
        
        # Average length ratio
        length_ratios = [len(pred) / max(len(ref), 1) for ref, pred in zip(references, predictions)]
        metrics['length_ratio'] = np.mean(length_ratios)
        
        return metrics
    
    def evaluate_model(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        print(f"🔍 Evaluating {model_name}...")
        
        references = []
        predictions = []
        
        for item in tqdm(self.test_data, desc=f"Evaluating {model_name}"):
            if isinstance(item, dict) and "text" in item:
                prompt = item["text"]
                
                # For this evaluation, we'll use the prompt as both input and expected output
                # In a real scenario, you'd have separate question-answer pairs
                expected = prompt
                
                try:
                    generated = self.generate_response(model, tokenizer, prompt)
                    references.append(expected)
                    predictions.append(generated)
                except Exception as e:
                    print(f"❌ Error generating response: {e}")
                    continue
        
        # Calculate metrics
        metrics = self.calculate_metrics(references, predictions)
        
        # Save detailed results
        results = []
        for ref, pred in zip(references, predictions):
            results.append({
                'reference': ref,
                'prediction': pred,
                'exact_match': ref.strip().lower() == pred.strip().lower()
            })
        
        return {
            'metrics': metrics,
            'results': results,
            'model_name': model_name
        }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation of all models"""
        print("🚀 Starting comprehensive evaluation...")
        
        # Load test data
        self.load_test_data()
        
        # Load baseline model
        self.load_baseline_model()
        
        # Evaluate baseline
        baseline_results = self.evaluate_model(
            self.baseline_model, 
            self.baseline_tokenizer, 
            "Baseline (Unfine-tuned)"
        )
        
        # Find and evaluate fine-tuned models
        results_dir = Path("results")
        fine_tuned_results = []
        
        if results_dir.exists():
            for run_dir in results_dir.iterdir():
                if run_dir.is_dir() and "run_" in run_dir.name:
                    peft_model_path = run_dir / "peft_model"
                    if peft_model_path.exists():
                        try:
                            self.load_fine_tuned_model(str(peft_model_path))
                            run_results = self.evaluate_model(
                                self.finetuned_model,
                                self.finetuned_tokenizer,
                                f"Fine-tuned ({run_dir.name})"
                            )
                            fine_tuned_results.append(run_results)
                        except Exception as e:
                            print(f"❌ Error evaluating {run_dir.name}: {e}")
                            continue
        
        # Compile all results
        all_results = [baseline_results] + fine_tuned_results
        
        # Save detailed results
        self.save_evaluation_results(all_results)
        
        # Print comparison
        self.print_comparison(all_results)
        
        return all_results
    
    def save_evaluation_results(self, results: List[Dict[str, Any]]):
        """Save evaluation results to files"""
        output_dir = Path("results/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary metrics
        summary = []
        for result in results:
            summary.append({
                'model_name': result['model_name'],
                **result['metrics']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
        
        # Save detailed results for each model
        for result in results:
            model_name = result['model_name'].replace(" ", "_").replace("(", "").replace(")", "")
            results_df = pd.DataFrame(result['results'])
            results_df.to_csv(output_dir / f"{model_name}_detailed_results.csv", index=False)
        
        # Save all results as JSON
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Evaluation results saved to {output_dir}/")
    
    def print_comparison(self, results: List[Dict[str, Any]]):
        """Print comparison of all models"""
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS COMPARISON")
        print("="*80)
        
        # Create comparison table
        metrics_to_show = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'exact_match', 'length_ratio']
        
        print(f"{'Model':<30}", end="")
        for metric in metrics_to_show:
            print(f"{metric:>10}", end="")
        print()
        
        print("-" * (30 + 10 * len(metrics_to_show)))
        
        for result in results:
            model_name = result['model_name'][:29]
            print(f"{model_name:<30}", end="")
            
            for metric in metrics_to_show:
                value = result['metrics'].get(metric, 0.0)
                if isinstance(value, float):
                    print(f"{value:>10.3f}", end="")
                else:
                    print(f"{value:>10}", end="")
            print()
        
        print("\n" + "="*80)
        
        # Find best model for each metric
        for metric in metrics_to_show:
            best_model = max(results, key=lambda x: x['metrics'].get(metric, 0))
            best_value = best_model['metrics'].get(metric, 0)
            print(f"🏆 Best {metric}: {best_model['model_name']} ({best_value:.3f})")

def main():
    """Main evaluation execution"""
    print("🎯 InterviewMate Comprehensive Evaluation")
    print("="*50)
    
    evaluator = InterviewMateEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to results/evaluation/")

if __name__ == "__main__":
    main()
