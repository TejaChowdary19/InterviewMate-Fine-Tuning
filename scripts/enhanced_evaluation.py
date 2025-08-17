#!/usr/bin/env python3
"""
Enhanced Evaluation Script for InterviewMate
Comprehensive evaluation of the enhanced model with expanded dataset
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInterviewMateEvaluator:
    def __init__(self, 
                 base_model_path: str = "tiiuae/falcon-rw-1b",
                 enhanced_model_path: str = "./results/enhanced_model"):
        
        self.base_model_path = base_model_path
        self.enhanced_model_path = enhanced_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1
        
        # Load datasets
        self.test_data = self.load_test_data()
        
    def load_test_data(self) -> List[Dict[str, str]]:
        """Load test data from enhanced dataset"""
        logger.info("Loading test data...")
        
        with open("data/enhanced_ai_engineer_dataset.json", "r") as f:
            full_data = json.load(f)
        
        # Use last 15% as test set (same split as training)
        test_size = int(len(full_data) * 0.15)
        test_data = full_data[-test_size:]
        
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
    
    def load_models(self) -> Tuple[Any, Any, Any, Any]:
        """Load both base and enhanced models"""
        logger.info("Loading models...")
        
        # Load base model
        base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Load enhanced model
        try:
            enhanced_tokenizer = AutoTokenizer.from_pretrained(self.enhanced_model_path)
            enhanced_model = PeftModel.from_pretrained(
                base_model, 
                self.enhanced_model_path
            )
            logger.info("Enhanced model loaded successfully")
        except Exception as e:
            logger.warning(f"Enhanced model not found, using base model: {e}")
            enhanced_tokenizer = base_tokenizer
            enhanced_model = base_model
        
        return base_tokenizer, base_model, enhanced_tokenizer, enhanced_model
    
    def generate_response(self, model, tokenizer, question: str, max_length: int = 512) -> str:
        """Generate response from model"""
        try:
            # Format input for instruction tuning
            input_text = f"""<INSTRUCTION>
You are an expert AI interview coach specializing in machine learning and data science. 
Provide comprehensive, accurate, and helpful answers to interview questions.
</INSTRUCTION>

<QUESTION>
{question}
</QUESTION>

<ANSWER>"""
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer part
            if "<ANSWER>" in response:
                answer_start = response.find("<ANSWER>") + len("<ANSWER>")
                response = response[answer_start:].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def calculate_rouge_scores(self, reference: str, prediction: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bleu_score(self, reference: str, prediction: str) -> float:
        """Calculate BLEU score"""
        try:
            reference_tokens = reference.split()
            prediction_tokens = prediction.split()
            
            # Calculate BLEU-4 score
            bleu_score = sentence_bleu([reference_tokens], prediction_tokens, 
                                     smoothing_function=self.smoothie)
            return bleu_score
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_exact_match(self, reference: str, prediction: str) -> float:
        """Calculate exact match score"""
        try:
            return 1.0 if reference.strip().lower() == prediction.strip().lower() else 0.0
        except Exception as e:
            logger.error(f"Error calculating exact match: {e}")
            return 0.0
    
    def calculate_length_ratio(self, reference: str, prediction: str) -> float:
        """Calculate length ratio between reference and prediction"""
        try:
            ref_length = len(reference.split())
            pred_length = len(prediction.split())
            
            if ref_length == 0:
                return 0.0
            
            return pred_length / ref_length
        except Exception as e:
            logger.error(f"Error calculating length ratio: {e}")
            return 0.0
    
    def evaluate_model(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        results = []
        total_metrics = {
            'rouge1': [], 'rouge2': [], 'rougeL': [],
            'bleu': [], 'exact_match': [], 'length_ratio': []
        }
        
        for i, item in enumerate(self.test_data):
            if i % 10 == 0:
                logger.info(f"Processing example {i+1}/{len(self.test_data)}")
            
            question = item['question']
            reference_answer = item['answer']
            
            # Generate response
            prediction = self.generate_response(model, tokenizer, question)
            
            # Calculate metrics
            rouge_scores = self.calculate_rouge_scores(reference_answer, prediction)
            bleu_score = self.calculate_bleu_score(reference_answer, prediction)
            exact_match = self.calculate_exact_match(reference_answer, prediction)
            length_ratio = self.calculate_length_ratio(reference_answer, prediction)
            
            # Store results
            result = {
                'question': question,
                'reference': reference_answer,
                'prediction': prediction,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'bleu': bleu_score,
                'exact_match': exact_match,
                'length_ratio': length_ratio
            }
            results.append(result)
            
            # Accumulate metrics
            for metric in total_metrics:
                total_metrics[metric].append(result[metric])
        
        # Calculate average metrics
        avg_metrics = {}
        for metric, values in total_metrics.items():
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
        
        # Calculate improvement percentages
        improvement_metrics = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu']:
            avg_key = f'avg_{metric}'
            if avg_key in avg_metrics:
                improvement_metrics[f'{metric}_improvement'] = avg_metrics[avg_key] * 100
        
        evaluation_results = {
            'model_name': model_name,
            'test_samples': len(self.test_data),
            'detailed_results': results,
            'average_metrics': avg_metrics,
            'improvement_metrics': improvement_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        return evaluation_results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of both models"""
        logger.info("🚀 Starting Comprehensive Evaluation...")
        
        # Load models
        base_tokenizer, base_model, enhanced_tokenizer, enhanced_model = self.load_models()
        
        # Evaluate base model
        logger.info("Evaluating base model...")
        base_results = self.evaluate_model(base_model, base_tokenizer, "Base Model")
        
        # Evaluate enhanced model
        logger.info("Evaluating enhanced model...")
        enhanced_results = self.evaluate_model(enhanced_model, enhanced_tokenizer, "Enhanced Model")
        
        # Compare results
        comparison = self.compare_models(base_results, enhanced_results)
        
        # Save results
        self.save_evaluation_results(base_results, enhanced_results, comparison)
        
        return {
            'base_results': base_results,
            'enhanced_results': enhanced_results,
            'comparison': comparison
        }
    
    def compare_models(self, base_results: Dict, enhanced_results: Dict) -> Dict[str, Any]:
        """Compare performance between base and enhanced models"""
        logger.info("Comparing model performance...")
        
        base_metrics = base_results['average_metrics']
        enhanced_metrics = enhanced_results['average_metrics']
        
        comparison = {}
        
        # Calculate improvements
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'exact_match']:
            base_key = f'avg_{metric}'
            enhanced_key = f'avg_{metric}'
            
            if base_key in base_metrics and enhanced_key in enhanced_metrics:
                base_val = base_metrics[base_key]
                enhanced_val = enhanced_metrics[enhanced_key]
                
                if base_val > 0:
                    improvement_pct = ((enhanced_val - base_val) / base_val) * 100
                else:
                    improvement_pct = 0.0
                
                comparison[f'{metric}_improvement_pct'] = improvement_pct
                comparison[f'{metric}_base'] = base_val
                comparison[f'{metric}_enhanced'] = enhanced_val
        
        # Overall assessment
        avg_improvement = np.mean([v for k, v in comparison.items() if 'improvement_pct' in k])
        comparison['overall_improvement_pct'] = avg_improvement
        
        # Performance ranking
        comparison['performance_ranking'] = "Enhanced Model" if avg_improvement > 0 else "Base Model"
        
        return comparison
    
    def save_evaluation_results(self, base_results: Dict, enhanced_results: Dict, comparison: Dict):
        """Save evaluation results to files"""
        logger.info("Saving evaluation results...")
        
        # Create results directory
        results_dir = Path("./results/enhanced_evaluation")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(results_dir / "base_model_evaluation.json", "w") as f:
            json.dump(base_results, f, indent=2)
        
        with open(results_dir / "enhanced_model_evaluation.json", "w") as f:
            json.dump(enhanced_results, f, indent=2)
        
        with open(results_dir / "model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for result in enhanced_results['detailed_results']:
            summary_data.append({
                'question': result['question'][:100] + "...",  # Truncate for CSV
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL'],
                'bleu': result['bleu'],
                'exact_match': result['exact_match'],
                'length_ratio': result['length_ratio']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(results_dir / "enhanced_evaluation_summary.csv", index=False)
        
        # Create performance summary
        performance_summary = {
            'evaluation_summary': {
                'base_model': {
                    'name': base_results['model_name'],
                    'test_samples': base_results['test_samples'],
                    'metrics': base_results['average_metrics']
                },
                'enhanced_model': {
                    'name': enhanced_results['model_name'],
                    'test_samples': enhanced_results['test_samples'],
                    'metrics': enhanced_results['average_metrics']
                },
                'comparison': comparison
            }
        }
        
        with open(results_dir / "performance_summary.json", "w") as f:
            json.dump(performance_summary, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        comparison = results['comparison']
        
        print("\n" + "="*80)
        print("🎯 ENHANCED INTERVIEWMATE EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n📊 Model Performance Comparison:")
        print(f"   Base Model Test Samples: {results['base_results']['test_samples']}")
        print(f"   Enhanced Model Test Samples: {results['enhanced_results']['test_samples']}")
        
        print(f"\n🚀 Performance Improvements:")
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'exact_match']:
            if f'{metric}_improvement_pct' in comparison:
                improvement = comparison[f'{metric}_improvement_pct']
                base_val = comparison[f'{metric}_base']
                enhanced_val = comparison[f'{metric}_enhanced']
                
                print(f"   {metric.upper()}: {base_val:.4f} → {enhanced_val:.4f} ({improvement:+.2f}%)")
        
        print(f"\n🏆 Overall Assessment:")
        print(f"   Average Improvement: {comparison['overall_improvement_pct']:+.2f}%")
        print(f"   Performance Ranking: {comparison['performance_ranking']}")
        
        print(f"\n📈 Dataset Enhancement Impact:")
        print(f"   Original Dataset: 302 examples")
        print(f"   Enhanced Dataset: 905 examples")
        print(f"   Dataset Expansion: +200%")
        
        print("\n" + "="*80)

def main():
    """Main function for enhanced evaluation"""
    logger.info("🚀 Starting Enhanced InterviewMate Evaluation...")
    
    # Create evaluator
    evaluator = EnhancedInterviewMateEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    logger.info("🎉 Enhanced evaluation completed!")
    logger.info("Check ./results/enhanced_evaluation/ for detailed results")

if __name__ == "__main__":
    main()

