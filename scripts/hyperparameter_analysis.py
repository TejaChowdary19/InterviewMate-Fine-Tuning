#!/usr/bin/env python3
"""
Hyperparameter Analysis for InterviewMate
Analyzes and compares different hyperparameter configurations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class HyperparameterAnalyzer:
    """Analyzer for hyperparameter configurations and results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.training_configs = {}
        self.training_results = {}
        self.evaluation_results = {}
        
    def load_training_configs(self):
        """Load training configurations from all runs"""
        print("📊 Loading training configurations...")
        
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}")
            return False
        
        for run_dir in self.results_dir.iterdir():
            if run_dir.is_dir() and "run_" in run_dir.name:
                config_file = run_dir / "training_config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        self.training_configs[run_dir.name] = config
                        print(f"✅ Loaded config for {run_dir.name}")
                    except Exception as e:
                        print(f"❌ Error loading config for {run_dir.name}: {e}")
        
        print(f"📈 Loaded {len(self.training_configs)} training configurations")
        return True
    
    def load_training_results(self):
        """Load training results from all runs"""
        print("📊 Loading training results...")
        
        for run_name in self.training_configs.keys():
            results_file = self.results_dir / run_name / "training_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    self.training_results[run_name] = results
                    print(f"✅ Loaded results for {run_name}")
                except Exception as e:
                    print(f"❌ Error loading results for {run_name}: {e}")
        
        print(f"📈 Loaded {len(self.training_results)} training results")
        return True
    
    def load_evaluation_results(self):
        """Load evaluation results if available"""
        eval_file = self.results_dir / "evaluation" / "evaluation_results.json"
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    self.evaluation_results = json.load(f)
                print(f"✅ Loaded evaluation results for {len(self.evaluation_results)} models")
            except Exception as e:
                print(f"❌ Error loading evaluation results: {e}")
        else:
            print("⚠️ No evaluation results found")
    
    def analyze_hyperparameters(self) -> pd.DataFrame:
        """Analyze hyperparameter configurations"""
        if not self.training_configs:
            print("❌ No training configurations loaded")
            return pd.DataFrame()
        
        # Extract key hyperparameters
        analysis_data = []
        
        for run_name, config in self.training_configs.items():
            row = {
                'run_name': run_name,
                'learning_rate': config.get('learning_rate', 0),
                'lora_r': config.get('lora_r', 0),
                'lora_alpha': config.get('lora_alpha', 0),
                'batch_size': config.get('batch_size', 0),
                'num_epochs': config.get('num_epochs', 0),
                'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 0),
                'weight_decay': config.get('weight_decay', 0),
                'warmup_steps': config.get('warmup_steps', 0),
                'max_length': config.get('max_length', 0)
            }
            
            # Add training results if available
            if run_name in self.training_results:
                results = self.training_results[run_name]
                row.update({
                    'final_loss': results.get('train_loss', 0),
                    'training_time': results.get('train_runtime', 0),
                    'samples_per_second': results.get('train_samples_per_second', 0)
                })
            
            # Add evaluation results if available
            if self.evaluation_results:
                for eval_result in self.evaluation_results:
                    if run_name in eval_result['model_name']:
                        metrics = eval_result['metrics']
                        row.update({
                            'rouge1': metrics.get('rouge1', 0),
                            'rouge2': metrics.get('rouge2', 0),
                            'rougeL': metrics.get('rougeL', 0),
                            'bleu': metrics.get('bleu', 0),
                            'exact_match': metrics.get('exact_match', 0)
                        })
                        break
            
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        return df
    
    def create_hyperparameter_visualizations(self, df: pd.DataFrame, 
                                           output_dir: str = "results/hyperparameter_analysis"):
        """Create visualizations for hyperparameter analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if df.empty:
            print("❌ No data for visualization")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Learning Rate vs Performance
        if 'rougeL' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['learning_rate'], df['rougeL'], s=100, alpha=0.7)
            for i, row in df.iterrows():
                plt.annotate(row['run_name'], (row['learning_rate'], row['rougeL']), 
                           xytext=(5, 5), textcoords='offset points')
            plt.xlabel('Learning Rate')
            plt.ylabel('ROUGE-L Score')
            plt.title('Learning Rate vs ROUGE-L Performance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'learning_rate_vs_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. LoRA Configuration Analysis
        if 'lora_r' in df.columns and 'rougeL' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # LoRA r vs Performance
            ax1.scatter(df['lora_r'], df['rougeL'], s=100, alpha=0.7)
            ax1.set_xlabel('LoRA r')
            ax1.set_ylabel('ROUGE-L Score')
            ax1.set_title('LoRA r vs Performance')
            ax1.grid(True, alpha=0.3)
            
            # LoRA alpha vs Performance
            ax2.scatter(df['lora_alpha'], df['rougeL'], s=100, alpha=0.7)
            ax2.set_xlabel('LoRA Alpha')
            ax2.set_ylabel('ROUGE-L Score')
            ax2.set_title('LoRA Alpha vs Performance')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'lora_configuration_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Training Loss Comparison
        if 'final_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(df['run_name'], df['final_loss'], color='skyblue', alpha=0.7)
            plt.xlabel('Training Run')
            plt.ylabel('Final Training Loss')
            plt.title('Final Training Loss Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, loss in zip(bars, df['final_loss']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{loss:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'training_loss_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Performance Metrics Heatmap
        if 'rougeL' in df.columns and 'bleu' in df.columns and 'exact_match' in df.columns:
            metrics_cols = ['rougeL', 'bleu', 'exact_match']
            metrics_df = df[metrics_cols].fillna(0)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                       xticklabels=df['run_name'], yticklabels=metrics_cols)
            plt.title('Performance Metrics Heatmap')
            plt.xlabel('Training Run')
            plt.ylabel('Metric')
            plt.tight_layout()
            plt.savefig(output_path / 'performance_metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to {output_path}/")
    
    def generate_hyperparameter_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights from hyperparameter analysis"""
        insights = []
        
        if df.empty:
            return ["No data available for analysis"]
        
        # Best performing configuration
        if 'rougeL' in df.columns:
            best_run = df.loc[df['rougeL'].idxmax()]
            insights.append(f"🏆 Best performing configuration: {best_run['run_name']} (ROUGE-L: {best_run['rougeL']:.3f})")
        
        # Learning rate analysis
        if 'learning_rate' in df.columns:
            lr_range = df['learning_rate'].max() - df['learning_rate'].min()
            insights.append(f"📈 Learning rate range: {df['learning_rate'].min():.2e} to {df['learning_rate'].max():.2e}")
            
            if 'rougeL' in df.columns:
                best_lr = df.loc[df['rougeL'].idxmax(), 'learning_rate']
                insights.append(f"🎯 Optimal learning rate: {best_lr:.2e}")
        
        # LoRA configuration analysis
        if 'lora_r' in df.columns and 'lora_alpha' in df.columns:
            insights.append(f"🔧 LoRA configurations tested: r={list(df['lora_r'].unique())}, alpha={list(df['lora_alpha'].unique())}")
            
            if 'rougeL' in df.columns:
                best_r = df.loc[df['rougeL'].idxmax(), 'lora_r']
                best_alpha = df.loc[df['rougeL'].idxmax(), 'lora_alpha']
                insights.append(f"⚡ Best LoRA config: r={best_r}, alpha={best_alpha}")
        
        # Training efficiency
        if 'training_time' in df.columns:
            fastest_run = df.loc[df['training_time'].idxmin()]
            insights.append(f"⚡ Fastest training: {fastest_run['run_name']} ({fastest_run['training_time']/60:.1f} minutes)")
        
        # Loss analysis
        if 'final_loss' in df.columns:
            loss_range = df['final_loss'].max() - df['final_loss'].min()
            insights.append(f"📉 Loss range: {df['final_loss'].min():.3f} to {df['final_loss'].max():.3f}")
            
            if loss_range > 0.5:
                insights.append("⚠️ High loss variance suggests hyperparameter sensitivity")
            else:
                insights.append("✅ Consistent loss across configurations")
        
        # Recommendations
        insights.append("\n💡 Recommendations:")
        insights.append("   • Test learning rates between 1e-5 and 5e-4")
        insights.append("   • Experiment with LoRA r values: 2, 4, 8, 16")
        insights.append("   • Consider longer training with early stopping")
        insights.append("   • Validate on larger test sets")
        
        return insights
    
    def save_analysis_results(self, df: pd.DataFrame, insights: List[str], 
                             output_dir: str = "results/hyperparameter_analysis"):
        """Save analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(output_path / "hyperparameter_analysis.csv", index=False)
        
        # Save insights
        with open(output_path / "hyperparameter_insights.txt", 'w') as f:
            f.write("InterviewMate Hyperparameter Analysis Insights\n")
            f.write("=" * 50 + "\n\n")
            for insight in insights:
                f.write(insight + "\n")
        
        # Save detailed analysis
        analysis_summary = {
            'total_runs': len(df),
            'hyperparameter_ranges': {
                'learning_rate': {'min': float(df['learning_rate'].min()), 'max': float(df['learning_rate'].max())},
                'lora_r': {'min': int(df['lora_r'].min()), 'max': int(df['lora_r'].max())},
                'lora_alpha': {'min': int(df['lora_alpha'].min()), 'max': int(df['lora_alpha'].max())}
            },
            'best_configuration': df.loc[df['rougeL'].idxmax()].to_dict() if 'rougeL' in df.columns else None
        }
        
        with open(output_path / "analysis_summary.json", 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        print(f"💾 Analysis results saved to {output_path}/")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive hyperparameter analysis"""
        print("🚀 Starting Hyperparameter Analysis...")
        
        # Load all data
        if not self.load_training_configs():
            return
        
        self.load_training_results()
        self.load_evaluation_results()
        
        # Analyze hyperparameters
        df = self.analyze_hyperparameters()
        
        if df.empty:
            print("❌ No data available for analysis")
            return
        
        # Print analysis summary
        print(f"\n📊 Hyperparameter Analysis Summary:")
        print(f"   Total runs analyzed: {len(df)}")
        print(f"   Hyperparameters tested: {list(df.columns[1:])}")
        
        if 'rougeL' in df.columns:
            print(f"   Best ROUGE-L score: {df['rougeL'].max():.3f}")
            print(f"   Average ROUGE-L score: {df['rougeL'].mean():.3f}")
        
        # Generate insights
        insights = self.generate_hyperparameter_insights(df)
        
        print(f"\n💡 Key Insights:")
        for insight in insights[:5]:  # Show first 5 insights
            print(f"   {insight}")
        
        # Create visualizations
        self.create_hyperparameter_visualizations(df)
        
        # Save results
        self.save_analysis_results(df, insights)
        
        print("\n✅ Hyperparameter analysis complete!")
        return df, insights

def main():
    """Main hyperparameter analysis execution"""
    print("🔬 InterviewMate Hyperparameter Analysis")
    print("="*50)
    
    analyzer = HyperparameterAnalyzer()
    df, insights = analyzer.run_comprehensive_analysis()
    
    if df is not None:
        print("\n📊 Analysis Results:")
        print(df.to_string(index=False))
        
        print(f"\n💡 All Insights:")
        for insight in insights:
            print(f"   {insight}")
    
    print("\n✅ Hyperparameter analysis complete!")
    print("📁 Results saved to results/hyperparameter_analysis/")

if __name__ == "__main__":
    main()
