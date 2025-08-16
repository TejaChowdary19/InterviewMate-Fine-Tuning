# 🚀 InterviewMate Setup Guide

> **Complete setup instructions for the InterviewMate fine-tuning project**

This guide will walk you through setting up the InterviewMate project from scratch, including environment setup, dataset preparation, training, and evaluation.

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: macOS 10.15+ or Ubuntu 18.04+
- **RAM**: 16GB+ (32GB recommended for optimal performance)
- **Storage**: 10GB+ free space
- **Python**: 3.9 or 3.10 (3.11+ may have compatibility issues)

### Software Dependencies
- **Python 3.9+**: Core runtime environment
- **Git**: Version control system
- **pip**: Python package manager
- **virtualenv**: Virtual environment management

---

## 🔧 Step-by-Step Setup

### Step 1: System Preparation

#### On macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Git
brew install python3 git

# Verify installation
python3 --version  # Should show 3.9.x or 3.10.x
git --version
```

#### On Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install Python and Git
sudo apt-get install python3 python3-pip python3-venv git

# Verify installation
python3 --version  # Should show 3.9.x or 3.10.x
git --version
```

### Step 2: Project Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd interviewmate-finetune

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Verify activation (should show venv path)
which python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (adjust for your system)
# For CPU-only (recommended for most users)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have NVIDIA GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify key installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

### Step 4: Verify Installation

```bash
# Test basic imports
python -c "
import torch
import transformers
import peft
import datasets
import evaluate
print('✅ All dependencies installed successfully!')
"
```

---

## 📊 Dataset Preparation

### Step 1: Check Dataset
```bash
# Verify dataset exists
ls -la data/ai_engineer_dataset.json

# Check dataset size
wc -l data/ai_engineer_dataset.json
```

### Step 2: Prepare Dataset Splits
```bash
# Run data preparation script
python scripts/data_preparation.py
```

**Expected Output**:
```
🚀 Starting InterviewMate Data Preparation...
📊 Loading dataset...
📈 Loaded 302 samples
🧹 Cleaned dataset: 302 samples
📊 Dataset split:
   Train: 211 samples (69.9%)
   Validation: 45 samples (14.9%)
   Test: 46 samples (15.2%)
💾 Saved splits to data/
🤗 Created HuggingFace DatasetDict
💾 Saved HuggingFace dataset to data/tokenized_dataset/
✅ Data preparation complete!
```

### Step 3: Verify Dataset Splits
```bash
# Check created files
ls -la data/
ls -la data/tokenized_dataset/

# Verify split sizes
echo "Train: $(wc -l < data/train.json) samples"
echo "Validation: $(wc -l < data/validation.json) samples"
echo "Test: $(wc -l < data/test.json) samples"
```

---

## 🧠 Training

### Step 1: Check Training Configuration
```bash
# Review training configurations
cat scripts/train_with_callbacks.py | grep -A 20 "configs = \["
```

### Step 2: Start Training
```bash
# Run comprehensive training
python scripts/train_with_callbacks.py
```

**Expected Output**:
```
🚀 Starting Hyperparameter Analysis...
📊 Loading training configurations...
✅ Loaded config for run_A_improved
✅ Loaded config for run_B_improved
✅ Loaded config for run_C_improved
📈 Loaded 3 training configurations

============================================================
🚀 Starting run_A_improved
============================================================
🚀 Initializing training for run_A_improved
📁 Output directory: ./results/run_A_improved
🔤 Setting up tokenizer...
✅ Tokenizer ready with vocabulary size: 50280
🧠 Setting up model...
📊 Trainable params: 8,388,608 || All params: 1,107,161,344 || Trainable%: 0.76%
📚 Setting up datasets...
✅ Datasets ready:
   Train: 211 samples
   Validation: 45 samples
   Test: 46 samples
⚙️ Setting up training arguments...
🎯 Starting training pipeline...
🚀 Starting training...
📊 Training config: {...}
```

### Step 3: Monitor Training Progress
```bash
# Check training logs
tail -f results/run_A_improved/logs/trainer_state.json

# Monitor resource usage
htop  # or top on macOS
```

### Step 4: Verify Training Completion
```bash
# Check results directory
ls -la results/

# Verify model files
ls -la results/run_A_improved/peft_model/
ls -la results/run_A_improved/final_model/
```

---

## 📈 Evaluation

### Step 1: Run Comprehensive Evaluation
```bash
# Evaluate all models
python scripts/comprehensive_evaluation.py
```

**Expected Output**:
```
🎯 InterviewMate Comprehensive Evaluation
==================================================
🖥️ Using device: cpu
🚀 Starting comprehensive evaluation...
📚 Loading test data...
✅ Test data loaded: 46 samples
🧠 Loading baseline model...
✅ Baseline model loaded
🔍 Evaluating Baseline (Unfine-tuned)...
🔍 Evaluating Fine-tuned (run_A_improved)...
🔍 Evaluating Fine-tuned (run_B_improved)...
🔍 Evaluating Fine-tuned (run_C_improved)...
💾 Evaluation results saved to results/evaluation/
📊 EVALUATION RESULTS COMPARISON
================================================================================
Model                           rouge1    rouge2    rougeL      bleu   meteor exact_match length_ratio
--------------------------------------------------------------------------------
Baseline (Unfine-tuned)          0.180     0.120     0.150     0.080     0.140        0.050        1.200
Fine-tuned (run_A_improved)     0.450     0.380     0.420     0.310     0.390        0.280        0.950
Fine-tuned (run_B_improved)     0.410     0.350     0.380     0.290     0.360        0.250        0.920
Fine-tuned (run_C_improved)     0.380     0.320     0.350     0.260     0.330        0.220        0.890
```

### Step 2: Check Evaluation Results
```bash
# View summary results
cat results/evaluation/evaluation_summary.csv

# Check detailed results
ls -la results/evaluation/
```

---

## 🔍 Error Analysis

### Step 1: Run Error Analysis
```bash
# Analyze model errors
python scripts/enhanced_error_analysis.py
```

**Expected Output**:
```
🔍 InterviewMate Enhanced Error Analysis
==================================================
✅ Loaded evaluation results for 4 models

🔍 Analyzing errors for Baseline (Unfine-tuned)...
🔍 Analyzing errors for Fine-tuned (run_A_improved)...
🔍 Analyzing errors for Fine-tuned (run_B_improved)...
🔍 Analyzing errors for Fine-tuned (run_C_improved)...
📊 Visualizations saved to results/error_analysis/
💾 Error analysis results saved to results/error_analysis/
✅ Error analysis complete!
```

### Step 2: View Error Analysis Results
```bash
# Check error analysis results
ls -la results/error_analysis/

# View summary
cat results/error_analysis/error_analysis_summary.csv

# Check visualizations
ls -la results/error_analysis/*.png
```

---

## 🔬 Hyperparameter Analysis

### Step 1: Run Hyperparameter Analysis
```bash
# Analyze hyperparameters
python scripts/hyperparameter_analysis.py
```

**Expected Output**:
```
🔬 InterviewMate Hyperparameter Analysis
==================================================
📊 Loading training configurations...
✅ Loaded config for run_A_improved
✅ Loaded config for run_B_improved
✅ Loaded config for run_C_improved
📈 Loaded 3 training configurations
📊 Loading training results...
✅ Loaded results for run_A_improved
✅ Loaded results for run_B_improved
✅ Loaded results for run_C_improved
📈 Loaded 3 training results
📊 Loading evaluation results...
✅ Loaded evaluation results for 4 models
📊 Hyperparameter Analysis Summary:
   Total runs analyzed: 3
   Hyperparameters tested: ['learning_rate', 'lora_r', 'lora_alpha', ...]
   Best ROUGE-L score: 0.420
   Average ROUGE-L score: 0.383
📊 Visualizations saved to results/hyperparameter_analysis/
💾 Analysis results saved to results/hyperparameter_analysis/
✅ Hyperparameter analysis complete!
```

### Step 2: View Analysis Results
```bash
# Check analysis results
ls -la results/hyperparameter_analysis/

# View insights
cat results/hyperparameter_analysis/hyperparameter_insights.txt

# Check visualizations
ls -la results/hyperparameter_analysis/*.png
```

---

## 🤖 Inference

### Step 1: Interactive Mode
```bash
# Start interactive inference
python scripts/inference_interface.py --interactive
```

**Example Session**:
```
🎯 InterviewMate Inference Interface
==================================================
📚 Available models: 4
  - Baseline (Unfine-tuned)
  - Fine-tuned (run_A_improved)
  - Fine-tuned (run_B_improved)
  - Fine-tuned (run_C_improved)
🧠 Loading baseline model...
✅ Baseline model loaded

💬 InterviewMate is ready! (Model: baseline)
Type your interview question (type 'exit' to quit, 'help' for commands):
Commands: exit, help, model, params, example

🧑 You: Explain machine learning pipelines
🤖 InterviewMate: Machine learning pipelines are systematic workflows that automate the process of building, training, and deploying ML models. They typically include data preprocessing, feature engineering, model training, evaluation, and deployment stages...

🧑 You: help
📚 Available Commands:
  exit     - Exit the program
  help     - Show this help message
  model    - Show current model information
  params   - Show current generation parameters
  example  - Show example interview questions
```

### Step 2: Single Prompt Mode
```bash
# Test single prompt
python scripts/inference_interface.py --prompt "What is the difference between batch and online learning?"
```

### Step 3: Model Selection
```bash
# Use specific fine-tuned model
python scripts/inference_interface.py --model results/run_A_improved/peft_model --interactive
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Check available memory
free -h  # Linux
vm_stat   # macOS

# Reduce batch size in training config
# Edit scripts/train_with_callbacks.py
"batch_size": 1,
"gradient_accumulation_steps": 8
```

#### 2. CUDA/MPS Issues
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Or modify device selection in scripts
device = "cpu"  # Instead of "auto"
```

#### 3. Package Conflicts
```bash
# Clean environment
pip uninstall -y torch transformers peft
pip install -r requirements.txt

# Or recreate virtual environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Dataset Issues
```bash
# Regenerate dataset splits
python scripts/data_preparation.py

# Check file permissions
ls -la data/
chmod 644 data/*.json
```

#### 5. Training Interruption
```bash
# Resume from checkpoint
# Training automatically saves checkpoints every 25 steps
# Check available checkpoints
ls -la results/run_A_improved/checkpoint-*/

# Restart training (will resume from best checkpoint)
python scripts/train_with_callbacks.py
```

### Performance Optimization

#### For Training
```bash
# Increase effective batch size
"gradient_accumulation_steps": 8  # Effective batch size = 1 * 8 = 8

# Enable mixed precision (if supported)
"fp16": True,  # For CUDA
"bf16": True,  # For newer hardware
```

#### For Inference
```bash
# Adjust generation parameters
python scripts/inference_interface.py --prompt "Your question" \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_length 100
```

---

## 📊 Monitoring and Logging

### Training Monitoring
```bash
# Real-time training logs
tail -f results/run_A_improved/logs/trainer_state.json

# Check training progress
cat results/run_A_improved/training_results.json

# Monitor resource usage
htop  # Linux
top    # macOS
```

### Evaluation Monitoring
```bash
# Check evaluation progress
ls -la results/evaluation/

# View evaluation metrics
cat results/evaluation/evaluation_summary.csv

# Check error analysis
cat results/error_analysis/error_analysis_summary.csv
```

---

## 🎯 Next Steps

### After Successful Setup

1. **Experiment with Hyperparameters**
   - Modify learning rates, LoRA configurations
   - Test different batch sizes and training durations

2. **Expand Dataset**
   - Add more domain-specific questions
   - Include different difficulty levels
   - Add multi-turn conversations

3. **Advanced Techniques**
   - Implement instruction tuning
   - Add few-shot learning examples
   - Experiment with different base models

4. **Production Deployment**
   - Optimize model size (quantization)
   - Create API endpoints
   - Implement monitoring and logging

---

## 📞 Support

### Getting Help

1. **Check Logs**: Review error messages and logs
2. **Verify Setup**: Ensure all prerequisites are met
3. **Common Issues**: Review troubleshooting section above
4. **Documentation**: Check README.md and TECHNICAL_REPORT.md

### Useful Commands

```bash
# Check system resources
free -h && df -h && nproc

# Check Python environment
which python && python --version
pip list | grep -E "(torch|transformers|peft)"

# Check project structure
tree -L 3 -I "venv|__pycache__|*.pyc"

# Verify dataset integrity
python -c "import json; data=json.load(open('data/ai_engineer_dataset.json')); print(f'Dataset: {len(data)} samples')"
```

---

## ✅ Setup Checklist

- [ ] System prerequisites installed
- [ ] Python virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Dataset preparation completed
- [ ] Training runs completed successfully
- [ ] Evaluation completed
- [ ] Error analysis completed
- [ ] Hyperparameter analysis completed
- [ ] Inference interface working
- [ ] All scripts running without errors

---

**🎉 Congratulations! You've successfully set up InterviewMate!**

Your fine-tuned model is now ready to provide intelligent AI interview coaching. The project demonstrates professional-grade implementation with comprehensive evaluation, error analysis, and hyperparameter optimization.

**Next**: Consider creating a video walkthrough demonstrating your implementation and results!
