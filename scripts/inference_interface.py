#!/usr/bin/env python3
"""
Improved Inference Interface for InterviewMate
Provides a functional interface for both baseline and fine-tuned models
"""

import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class InterviewMateInference:
    """Improved inference interface for InterviewMate models"""
    
    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model = None
        self.current_tokenizer = None
        self.model_type = None
        
        print(f"🖥️ Using device: {self.device}")
        
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all available models for inference"""
        available_models = [
            {
                'name': 'Baseline (Unfine-tuned)',
                'path': 'baseline',
                'type': 'baseline'
            }
        ]
        
        # Find fine-tuned models
        results_dir = Path("results")
        if results_dir.exists():
            for run_dir in results_dir.iterdir():
                if run_dir.is_dir() and "run_" in run_dir.name:
                    peft_model_path = run_dir / "peft_model"
                    if peft_model_path.exists():
                        available_models.append({
                            'name': f'Fine-tuned ({run_dir.name})',
                            'path': str(peft_model_path),
                            'type': 'finetuned'
                        })
        
        return available_models
    
    def load_baseline_model(self):
        """Load the baseline model"""
        print("🧠 Loading baseline model...")
        
        self.current_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.current_tokenizer.pad_token is None:
            self.current_tokenizer.add_special_tokens({'pad_token': self.current_tokenizer.eos_token})
        
        self.current_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.current_model.eval()
        
        self.model_type = "baseline"
        print("✅ Baseline model loaded")
        
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        print(f"🎯 Loading fine-tuned model from {model_path}...")
        
        # Load tokenizer
        if Path(model_path).exists():
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.current_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.add_special_tokens({'pad_token': self.current_tokenizer.eos_token})
        
        # Load base model
        self.current_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load LoRA weights
        self.current_model = PeftModel.from_pretrained(self.current_model, model_path)
        self.current_model.eval()
        
        self.model_type = "finetuned"
        print("✅ Fine-tuned model loaded")
    
    def load_model(self, model_path: str = "baseline"):
        """Load a specific model"""
        if model_path == "baseline":
            self.load_baseline_model()
        else:
            self.load_fine_tuned_model(model_path)
    
    def generate_response(self, prompt: str, 
                         max_length: int = 100,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         do_sample: bool = True) -> str:
        """Generate response from the loaded model"""
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Prepare input
        inputs = self.current_tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = self.current_model.generate(
                inputs,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.current_tokenizer.eos_token_id,
                eos_token_id=self.current_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def interactive_mode(self):
        """Run interactive inference mode"""
        if self.current_model is None:
            print("❌ No model loaded. Loading baseline model...")
            self.load_baseline_model()
        
        print(f"\n💬 InterviewMate is ready! (Model: {self.model_type})")
        print("Type your interview question (type 'exit' to quit, 'help' for commands):")
        print("Commands: exit, help, model, params, example")
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Exiting InterviewMate. Good luck with your prep!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'model':
                    self.show_model_info()
                elif user_input.lower() == 'params':
                    self.show_generation_params()
                elif user_input.lower() == 'example':
                    self.show_example_questions()
                elif user_input.lower() == '':
                    continue
                else:
                    # Generate response
                    print("🤖 InterviewMate: ", end="", flush=True)
                    response = self.generate_response(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting InterviewMate. Good luck with your prep!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show available commands"""
        print("\n📚 Available Commands:")
        print("  exit     - Exit the program")
        print("  help     - Show this help message")
        print("  model    - Show current model information")
        print("  params   - Show current generation parameters")
        print("  example  - Show example interview questions")
    
    def show_model_info(self):
        """Show current model information"""
        if self.current_model is None:
            print("❌ No model loaded")
            return
        
        print(f"\n🧠 Current Model Information:")
        print(f"  Type: {self.model_type}")
        print(f"  Base Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.current_model.parameters()):,}")
        
        if self.model_type == "finetuned":
            trainable_params = sum(p.numel() for p in self.current_model.parameters() if p.requires_grad)
            print(f"  Trainable Parameters: {trainable_params:,}")
    
    def show_generation_params(self):
        """Show current generation parameters"""
        print("\n⚙️ Current Generation Parameters:")
        print("  max_length: 100")
        print("  temperature: 0.7")
        print("  top_p: 0.9")
        print("  top_k: 50")
        print("  do_sample: True")
        print("  repetition_penalty: 1.1")
    
    def show_example_questions(self):
        """Show example interview questions"""
        examples = [
            "Explain how you would design a scalable machine learning system for real-time fraud detection.",
            "What is the difference between batch normalization and layer normalization?",
            "How do you choose between a decision tree, a random forest, and a gradient boosting machine?",
            "Describe a situation where you had to debug a model that was underperforming in production.",
            "How do you optimize inference speed in a deep learning model?"
        ]
        
        print("\n💡 Example Interview Questions:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
    
    def batch_inference(self, prompts: List[str], output_file: Optional[str] = None) -> List[Dict[str, str]]:
        """Run batch inference on multiple prompts"""
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        results = []
        
        print(f"🚀 Running batch inference on {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing {i}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                response = self.generate_response(prompt)
                results.append({
                    'prompt': prompt,
                    'response': response,
                    'model_type': self.model_type
                })
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {e}")
                results.append({
                    'prompt': prompt,
                    'response': f"ERROR: {e}",
                    'model_type': self.model_type
                })
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")
        
        return results

def main():
    """Main inference execution"""
    parser = argparse.ArgumentParser(description="InterviewMate Inference Interface")
    parser.add_argument("--model", type=str, default="baseline", 
                       help="Model to load (baseline or path to fine-tuned model)")
    parser.add_argument("--prompt", type=str, help="Single prompt for inference")
    parser.add_argument("--batch", type=str, help="File containing prompts for batch inference")
    parser.add_argument("--output", type=str, help="Output file for batch results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    print("🎯 InterviewMate Inference Interface")
    print("="*50)
    
    # Initialize inference interface
    inference = InterviewMateInference()
    
    # List available models
    available_models = inference.list_available_models()
    print(f"📚 Available models: {len(available_models)}")
    for model in available_models:
        print(f"  - {model['name']}")
    
    # Load specified model
    try:
        inference.load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run inference based on arguments
    if args.prompt:
        # Single prompt inference
        print(f"\n💬 Prompt: {args.prompt}")
        response = inference.generate_response(args.prompt)
        print(f"🤖 Response: {response}")
        
    elif args.batch:
        # Batch inference
        try:
            with open(args.batch, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            results = inference.batch_inference(prompts, args.output)
            print(f"✅ Batch inference complete: {len(results)} results")
            
        except Exception as e:
            print(f"❌ Error in batch inference: {e}")
            
    elif args.interactive:
        # Interactive mode
        inference.interactive_mode()
        
    else:
        # Default to interactive mode
        print("💡 No specific mode specified. Running in interactive mode...")
        inference.interactive_mode()

if __name__ == "__main__":
    main()
