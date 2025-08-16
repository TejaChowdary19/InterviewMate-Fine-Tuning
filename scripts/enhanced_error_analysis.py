#!/usr/bin/env python3
"""
Enhanced Error Analysis for InterviewMate
Provides detailed error categorization, pattern identification, and improvement suggestions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedErrorAnalyzer:
    """Enhanced error analyzer for InterviewMate models"""
    
    def __init__(self, evaluation_results_path: str = "results/evaluation/evaluation_results.json"):
        self.results_path = Path(evaluation_results_path)
        self.results = None
        self.error_categories = {
            'hallucination': 'Model generates incorrect or made-up information',
            'incomplete': 'Model provides partial or incomplete responses',
            'off_topic': 'Model responds to something other than the question',
            'repetitive': 'Model repeats the same information multiple times',
            'generic': 'Model gives overly general, non-specific responses',
            'technical_error': 'Model makes technical mistakes or confuses concepts',
            'length_mismatch': 'Response length is significantly different from expected',
            'format_error': 'Model doesn\'t follow expected response format'
        }
        
    def load_results(self):
        """Load evaluation results"""
        if not self.results_path.exists():
            print(f"❌ Evaluation results not found at {self.results_path}")
            print("Please run comprehensive_evaluation.py first")
            return False
        
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Loaded evaluation results for {len(self.results)} models")
        return True
    
    def categorize_errors(self, reference: str, prediction: str) -> List[str]:
        """Categorize errors in a prediction"""
        categories = []
        
        # Length mismatch
        ref_len = len(reference.split())
        pred_len = len(prediction.split())
        if pred_len < ref_len * 0.5 or pred_len > ref_len * 2.0:
            categories.append('length_mismatch')
        
        # Generic responses
        generic_phrases = [
            'be confident', 'communicate clearly', 'practice regularly',
            'stay positive', 'be yourself', 'prepare well',
            'show enthusiasm', 'demonstrate skills', 'be professional'
        ]
        if any(phrase in prediction.lower() for phrase in generic_phrases):
            categories.append('generic')
        
        # Repetitive content
        words = prediction.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            if any(count > 3 for count in word_counts.values()):
                categories.append('repetitive')
        
        # Technical errors (basic checks)
        technical_terms = ['spark', 'hadoop', 'database', 'pipeline', 'ml', 'ai']
        if any(term in reference.lower() for term in technical_terms):
            if not any(term in prediction.lower() for term in technical_terms):
                categories.append('technical_error')
        
        # Hallucination (if prediction is too different from reference)
        ref_words = set(reference.lower().split())
        pred_words = set(prediction.lower().split())
        overlap = len(ref_words.intersection(pred_words))
        if len(ref_words) > 0 and overlap / len(ref_words) < 0.3:
            categories.append('hallucination')
        
        # Incomplete responses
        if pred_len < ref_len * 0.7:
            categories.append('incomplete')
        
        # Off-topic (if no relevant keywords overlap)
        if overlap == 0 and len(ref_words) > 5:
            categories.append('off_topic')
        
        return categories if categories else ['acceptable']
    
    def analyze_model_errors(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze errors for a specific model"""
        model_name = model_results['model_name']
        results = model_results['results']
        
        print(f"\n🔍 Analyzing errors for {model_name}...")
        
        # Categorize all errors
        error_analysis = defaultdict(list)
        error_counts = defaultdict(int)
        
        for result in results:
            reference = result['reference']
            prediction = result['prediction']
            exact_match = result['exact_match']
            
            if not exact_match:
                error_categories = self.categorize_errors(reference, prediction)
                for category in error_categories:
                    error_analysis[category].append({
                        'reference': reference,
                        'prediction': prediction,
                        'reference_length': len(reference.split()),
                        'prediction_length': len(prediction.split())
                    })
                    error_counts[category] += 1
        
        # Calculate error statistics
        total_samples = len(results)
        error_rate = (total_samples - sum(1 for r in results if r['exact_match'])) / total_samples
        
        return {
            'model_name': model_name,
            'total_samples': total_samples,
            'error_rate': error_rate,
            'error_categories': dict(error_counts),
            'error_analysis': dict(error_analysis),
            'error_distribution': {cat: count/total_samples for cat, count in error_counts.items()}
        }
    
    def identify_error_patterns(self, error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in errors"""
        patterns = {}
        
        for category, errors in error_analysis['error_analysis'].items():
            if category == 'acceptable':
                continue
                
            category_patterns = {
                'avg_reference_length': np.mean([e['reference_length'] for e in errors]),
                'avg_prediction_length': np.mean([e['prediction_length'] for e in errors]),
                'length_ratio': np.mean([e['prediction_length'] / max(e['reference_length'], 1) for e in errors]),
                'common_words': Counter(),
                'sample_errors': errors[:3]  # First 3 examples
            }
            
            # Analyze common words in problematic responses
            for error in errors:
                words = error['prediction'].lower().split()
                category_patterns['common_words'].update(words)
            
            patterns[category] = category_patterns
        
        return patterns
    
    def generate_improvement_suggestions(self, error_analysis: Dict[str, Any], 
                                      patterns: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on error analysis"""
        suggestions = []
        
        # General suggestions
        if error_analysis['error_rate'] > 0.5:
            suggestions.append("🔴 High error rate detected. Consider increasing training data or adjusting hyperparameters.")
        
        # Category-specific suggestions
        if 'hallucination' in error_analysis['error_categories']:
            suggestions.append("🧠 Hallucination detected. Consider adding more diverse training examples and implementing fact-checking.")
        
        if 'generic' in error_analysis['error_categories']:
            suggestions.append("📝 Generic responses detected. Add more specific, technical examples to training data.")
        
        if 'technical_error' in error_analysis['error_categories']:
            suggestions.append("⚙️ Technical errors detected. Validate training data accuracy and add domain-specific examples.")
        
        if 'length_mismatch' in error_analysis['error_categories']:
            suggestions.append("📏 Length mismatch detected. Consider adjusting max_length parameter or adding length constraints.")
        
        if 'repetitive' in error_analysis['error_categories']:
            suggestions.append("🔄 Repetitive content detected. Implement repetition penalty in generation parameters.")
        
        # Training suggestions
        suggestions.append("📚 Increase training data diversity with more edge cases and complex scenarios.")
        suggestions.append("🎯 Implement few-shot learning with high-quality examples.")
        suggestions.append("⚡ Consider using larger base models or more sophisticated fine-tuning techniques.")
        
        return suggestions
    
    def create_error_visualizations(self, all_analyses: List[Dict[str, Any]], 
                                  output_dir: str = "results/error_analysis"):
        """Create visualizations for error analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Error rate comparison
        plt.figure(figsize=(12, 6))
        models = [analysis['model_name'] for analysis in all_analyses]
        error_rates = [analysis['error_rate'] for analysis in all_analyses]
        
        plt.bar(models, error_rates, color=['red' if rate > 0.5 else 'orange' if rate > 0.3 else 'green' for rate in error_rates])
        plt.title('Error Rate Comparison Across Models')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'error_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error category distribution for each model
        for analysis in all_analyses:
            if 'error_categories' in analysis:
                plt.figure(figsize=(10, 6))
                categories = list(analysis['error_categories'].keys())
                counts = list(analysis['error_categories'].values())
                
                plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
                plt.title(f'Error Distribution - {analysis["model_name"]}')
                plt.tight_layout()
                plt.savefig(output_path / f'error_distribution_{analysis["model_name"].replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📊 Visualizations saved to {output_path}/")
    
    def run_comprehensive_error_analysis(self):
        """Run comprehensive error analysis"""
        print("🚀 Starting Enhanced Error Analysis...")
        
        if not self.load_results():
            return
        
        all_analyses = []
        all_patterns = {}
        
        # Analyze each model
        for model_results in self.results:
            analysis = self.analyze_model_errors(model_results)
            patterns = self.identify_error_patterns(analysis)
            
            all_analyses.append(analysis)
            all_patterns[analysis['model_name']] = patterns
            
            # Generate suggestions
            suggestions = self.generate_improvement_suggestions(analysis, patterns)
            
            # Print analysis summary
            print(f"\n📊 Error Analysis Summary for {analysis['model_name']}:")
            print(f"   Total samples: {analysis['total_samples']}")
            print(f"   Error rate: {analysis['error_rate']:.2%}")
            print(f"   Error categories: {dict(analysis['error_categories'])}")
            
            print(f"\n💡 Improvement Suggestions:")
            for suggestion in suggestions:
                print(f"   {suggestion}")
        
        # Create visualizations
        self.create_error_visualizations(all_analyses)
        
        # Save detailed analysis
        self.save_error_analysis(all_analyses, all_patterns)
        
        print("\n✅ Error analysis complete!")
        return all_analyses, all_patterns
    
    def save_error_analysis(self, analyses: List[Dict[str, Any]], 
                           patterns: Dict[str, Any]):
        """Save error analysis results"""
        output_dir = Path("results/error_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = []
        for analysis in analyses:
            summary.append({
                'model_name': analysis['model_name'],
                'total_samples': analysis['total_samples'],
                'error_rate': analysis['error_rate'],
                'error_categories': analysis['error_categories']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_dir / "error_analysis_summary.csv", index=False)
        
        # Save detailed analysis
        with open(output_dir / "detailed_error_analysis.json", 'w') as f:
            json.dump({
                'analyses': analyses,
                'patterns': patterns
            }, f, indent=2, default=str)
        
        print(f"💾 Error analysis results saved to {output_dir}/")

def main():
    """Main error analysis execution"""
    print("🔍 InterviewMate Enhanced Error Analysis")
    print("="*50)
    
    analyzer = EnhancedErrorAnalyzer()
    analyses, patterns = analyzer.run_comprehensive_error_analysis()
    
    print("\n✅ Error analysis complete!")
    print("📁 Results saved to results/error_analysis/")

if __name__ == "__main__":
    main()
