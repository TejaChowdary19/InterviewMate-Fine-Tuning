
# 🤖 InterviewMate: Fine-Tuning Falcon for Domain-Specific Interview Coaching

> 📅 Last updated: 2025-07-08 02:16:16

---

## 📘 Table of Contents

- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Model Selection](#model-selection)
- [Fine-Tuning Setup](#fine-tuning-setup)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Model Evaluation](#model-evaluation)
- [Error Analysis](#error-analysis)
- [Inference Pipeline](#inference-pipeline)
- [Video Walkthrough & Documentation](#video-walkthrough--documentation)
- [Environment Setup](#environment-setup)

---

## 🧠 Overview

**InterviewMate** is a domain-specific interview coaching assistant, fine-tuned using **LoRA** on the **Falcon-RW-1B** language model. It is designed to simulate technical interview scenarios, especially for data engineering roles.

---

## 📂 Dataset Preparation

- **Source**: Custom-created JSON file with `"text"` field.
- **Format**: Each entry contains a QA-style prompt for interview.
- **Cleaning**: Removed noise, ensured uniform formatting.
- **Splits**:
  - Training: `data/ai_engineer_dataset.json`
  - Testing: `evaluation_data.json`
- **Tokenization**: `max_length=256`, truncation & padding enabled.

---

## 🏗️ Model Selection

- **Model**: [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b)
  - Lightweight (Mac-compatible)
  - Instruction-following capability
- **LoRA Adaptation**:
  - `target_modules=["query_key_value"]`
  - `r=8`, `lora_alpha=16`, `lora_dropout=0.05`

---

## 🛠️ Fine-Tuning Setup

- **Trainer API** used with:
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=4`
  - `num_train_epochs=5`
- **Checkpointing**: Every 10 steps
- **Device**: CPU (MPS not stable for Falcon)
- **Collator**: `DataCollatorForLanguageModeling(mlm=False)`

---

## 🔬 Hyperparameter Optimization

- **Search Strategy**: Manual grid search
- **Tested Configurations**:
  1. `lr=2e-4`, `epochs=5`, `r=8` ✅ *(selected best)*
  2. `lr=1e-4`, `epochs=5`, `r=4`
  3. `lr=2e-4`, `epochs=3`, `r=8`
- **Selection Criteria**: ROUGE-L score & qualitative inspection

(venv) tejachowdary@Tejas-Mac interviewmate-finetune % python scripts/hyperparameter_search.py
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
  warnings.warn(

===== Starting run_A =====

/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/bitsandbytes/cextension.py:34: UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
  warn("The installed version of bitsandbytes was compiled without GPU support. "
'NoneType' object has no attribute 'cadam32bit_grad_fp32'
/Users/tejachowdary/interviewmate-finetune/scripts/hyperparameter_search.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                 | 0/75 [00:00<?, ?it/s]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 3.0122, 'grad_norm': 1.8744843006134033, 'learning_rate': 0.00018933333333333335, 'epoch': 0.2}     
{'loss': 2.5762, 'grad_norm': 2.5482139587402344, 'learning_rate': 0.00017600000000000002, 'epoch': 0.4}     
 13%|█████████▌                                                              | 10/75 [00:32<03:30,  3.24s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.1758, 'grad_norm': 2.4097938537597656, 'learning_rate': 0.00016266666666666667, 'epoch': 0.6}     
{'loss': 1.4934, 'grad_norm': 1.9616583585739136, 'learning_rate': 0.00014933333333333335, 'epoch': 0.8}     
 27%|███████████████████▏                                                    | 20/75 [01:06<02:56,  3.22s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 1.3773, 'grad_norm': 2.4755005836486816, 'learning_rate': 0.00013600000000000003, 'epoch': 1.0}     
 33%|████████████████████████                                                | 25/75 [01:24<02:50,  3.40s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.9993, 'grad_norm': 2.3572838306427, 'learning_rate': 0.00012266666666666668, 'epoch': 1.2}        
 40%|████████████████████████████▊                                           | 30/75 [01:40<02:26,  3.26s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 1.0783, 'grad_norm': 6.08904504776001, 'learning_rate': 0.00010933333333333333, 'epoch': 1.4}       
{'loss': 0.8867, 'grad_norm': 2.5320794582366943, 'learning_rate': 9.6e-05, 'epoch': 1.6}                    
 53%|██████████████████████████████████████▍                                 | 40/75 [02:15<02:01,  3.48s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.8431, 'grad_norm': 2.7483320236206055, 'learning_rate': 8.266666666666667e-05, 'epoch': 1.8}      
{'loss': 0.6863, 'grad_norm': 2.390606164932251, 'learning_rate': 6.933333333333334e-05, 'epoch': 2.0}       
 67%|████████████████████████████████████████████████                        | 50/75 [02:58<01:56,  4.66s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 0.61, 'grad_norm': 2.6772403717041016, 'learning_rate': 5.6000000000000006e-05, 'epoch': 2.2}       
{'loss': 0.5019, 'grad_norm': 2.4624760150909424, 'learning_rate': 4.266666666666667e-05, 'epoch': 2.4}      
 80%|█████████████████████████████████████████████████████████▌              | 60/75 [03:50<01:15,  5.03s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.3967, 'grad_norm': 3.521871328353882, 'learning_rate': 2.9333333333333336e-05, 'epoch': 2.6}      
{'loss': 0.3973, 'grad_norm': 2.3116343021392822, 'learning_rate': 1.6000000000000003e-05, 'epoch': 2.8}     
 93%|███████████████████████████████████████████████████████████████████▏    | 70/75 [04:37<00:22,  4.52s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 0.4515, 'grad_norm': 4.173069477081299, 'learning_rate': 2.666666666666667e-06, 'epoch': 3.0}       
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:01<00:00,  4.66s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'train_runtime': 303.7093, 'train_samples_per_second': 0.988, 'train_steps_per_second': 0.247, 'train_loss': 1.1657305240631104, 'epoch': 3.0}
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:03<00:00,  4.05s/it]
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(

===== Starting run_B =====

/Users/tejachowdary/interviewmate-finetune/scripts/hyperparameter_search.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                 | 0/75 [00:00<?, ?it/s]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 3.1184, 'grad_norm': 1.1946017742156982, 'learning_rate': 9.466666666666667e-05, 'epoch': 0.2}      
{'loss': 3.0888, 'grad_norm': 1.558526873588562, 'learning_rate': 8.800000000000001e-05, 'epoch': 0.4}       
{'loss': 2.9938, 'grad_norm': 1.9344531297683716, 'learning_rate': 8.133333333333334e-05, 'epoch': 0.6}      
{'loss': 2.6287, 'grad_norm': 2.251981496810913, 'learning_rate': 7.466666666666667e-05, 'epoch': 0.8}       
 27%|███████████████████▏                                                    | 20/75 [01:29<03:40,  4.00s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.57, 'grad_norm': 2.104883909225464, 'learning_rate': 6.800000000000001e-05, 'epoch': 1.0}         
 33%|████████████████████████                                                | 25/75 [01:52<03:35,  4.30s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.349, 'grad_norm': 2.5524072647094727, 'learning_rate': 6.133333333333334e-05, 'epoch': 1.2}       
 40%|████████████████████████████▊                                           | 30/75 [02:12<03:16,  4.37s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.2802, 'grad_norm': 2.2798891067504883, 'learning_rate': 5.466666666666666e-05, 'epoch': 1.4}      
{'loss': 2.2767, 'grad_norm': 2.782996416091919, 'learning_rate': 4.8e-05, 'epoch': 1.6}                     
 53%|██████████████████████████████████████▍                                 | 40/75 [03:00<02:43,  4.68s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.1515, 'grad_norm': 2.6125521659851074, 'learning_rate': 4.133333333333333e-05, 'epoch': 1.8}      
{'loss': 1.8809, 'grad_norm': 2.8465993404388428, 'learning_rate': 3.466666666666667e-05, 'epoch': 2.0}      
 67%|████████████████████████████████████████████████                        | 50/75 [03:49<02:00,  4.80s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 1.9371, 'grad_norm': 2.8415706157684326, 'learning_rate': 2.8000000000000003e-05, 'epoch': 2.2}     
{'loss': 1.731, 'grad_norm': 2.6564226150512695, 'learning_rate': 2.1333333333333335e-05, 'epoch': 2.4}      
 80%|█████████████████████████████████████████████████████████▌              | 60/75 [04:34<01:04,  4.29s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 1.6436, 'grad_norm': 2.3705623149871826, 'learning_rate': 1.4666666666666668e-05, 'epoch': 2.6}     
{'loss': 1.5848, 'grad_norm': 2.108274459838867, 'learning_rate': 8.000000000000001e-06, 'epoch': 2.8}       
 93%|███████████████████████████████████████████████████████████████████▏    | 70/75 [05:18<00:21,  4.24s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 1.6277, 'grad_norm': 2.2567837238311768, 'learning_rate': 1.3333333333333334e-06, 'epoch': 3.0}     
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:40<00:00,  4.46s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'train_runtime': 341.9684, 'train_samples_per_second': 0.877, 'train_steps_per_second': 0.219, 'train_loss': 2.2574768511454266, 'epoch': 3.0}
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:41<00:00,  4.56s/it]
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(

===== Starting run_C =====

/Users/tejachowdary/interviewmate-finetune/scripts/hyperparameter_search.py:65: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
  0%|                                                                                 | 0/75 [00:00<?, ?it/s]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 3.1449, 'grad_norm': 0.751812756061554, 'learning_rate': 4.7333333333333336e-05, 'epoch': 0.2}      
{'loss': 3.2137, 'grad_norm': 0.7988368272781372, 'learning_rate': 4.4000000000000006e-05, 'epoch': 0.4}     
{'loss': 3.2118, 'grad_norm': 0.9981998205184937, 'learning_rate': 4.066666666666667e-05, 'epoch': 0.6}      
{'loss': 2.9717, 'grad_norm': 1.248125433921814, 'learning_rate': 3.733333333333334e-05, 'epoch': 0.8}       
 27%|███████████████████▏                                                    | 20/75 [01:32<04:32,  4.96s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.9588, 'grad_norm': 1.1402063369750977, 'learning_rate': 3.4000000000000007e-05, 'epoch': 1.0}     
 33%|████████████████████████                                                | 25/75 [01:57<04:06,  4.92s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.9409, 'grad_norm': 1.467175841331482, 'learning_rate': 3.066666666666667e-05, 'epoch': 1.2}       
 40%|████████████████████████████▊                                           | 30/75 [02:20<03:25,  4.56s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.9252, 'grad_norm': 1.436004400253296, 'learning_rate': 2.733333333333333e-05, 'epoch': 1.4}       
{'loss': 3.0644, 'grad_norm': 1.5957622528076172, 'learning_rate': 2.4e-05, 'epoch': 1.6}                    
 53%|██████████████████████████████████████▍                                 | 40/75 [03:03<02:34,  4.43s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 3.0335, 'grad_norm': 1.6648900508880615, 'learning_rate': 2.0666666666666666e-05, 'epoch': 1.8}     
{'loss': 2.8175, 'grad_norm': 1.823523998260498, 'learning_rate': 1.7333333333333336e-05, 'epoch': 2.0}      
 67%|████████████████████████████████████████████████                        | 50/75 [03:48<01:53,  4.53s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
{'loss': 2.9689, 'grad_norm': 1.7594366073608398, 'learning_rate': 1.4000000000000001e-05, 'epoch': 2.2}     
{'loss': 2.9333, 'grad_norm': 1.9299172163009644, 'learning_rate': 1.0666666666666667e-05, 'epoch': 2.4}     
 80%|█████████████████████████████████████████████████████████▌              | 60/75 [04:33<01:06,  4.41s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.7427, 'grad_norm': 1.8641496896743774, 'learning_rate': 7.333333333333334e-06, 'epoch': 2.6}      
{'loss': 2.811, 'grad_norm': 1.8617912530899048, 'learning_rate': 4.000000000000001e-06, 'epoch': 2.8}       
 93%|███████████████████████████████████████████████████████████████████▏    | 70/75 [05:19<00:22,  4.47s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'loss': 2.819, 'grad_norm': 1.7143319845199585, 'learning_rate': 6.666666666666667e-07, 'epoch': 3.0}       
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:44<00:00,  4.86s/it]/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(
{'train_runtime': 345.928, 'train_samples_per_second': 0.867, 'train_steps_per_second': 0.217, 'train_loss': 2.970481732686361, 'epoch': 3.0}
100%|████████████████████████████████████████████████████████████████████████| 75/75 [05:45<00:00,  4.61s/it]
/Users/tejachowdary/interviewmate-finetune/venv/lib/python3.9/site-packages/peft/utils/save_and_load.py:252: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
  warnings.warn(

---

## 📈 Model Evaluation

- **Metric**: `rouge` (via 🤗 `evaluate` lib)
- **Test Data**: 4 manually created interview-style prompts
- **Comparison**:
  - **Base Falcon**: Generic responses, lacks focus
  - **LoRA Fine-tuned**: More relevant, domain-aware

---

## ❌ Error Analysis

### 🧪 Observed Issues:
1. **Generic Outputs**:
   - **Prompt**: `"Give me tips for a data engineering interview."`
   - **Output**: `"Be confident, communicate clearly."`
   - 🔎 *Too generic — missing technical specifics.*

2. **Concept Confusion**:
   - **Prompt**: `"Differences between Spark and Hadoop?"`
   - **Output**: `"They are both databases used for big data."`
   - ❗ *Wrong classification — they are processing engines.*

### 🔁 Patterns Identified:
- Hallucination when terms are ambiguous.
- Over-simplification of complex prompts.

### 🛠 Suggested Fixes:
- Increase dataset size with more nuanced questions.
- Add targeted few-shot examples in prompts.

---

## ⚙️ Inference Pipeline

- **Script**: `inference.py`
- **Usage**:

```bash
python inference.py --prompt "How do you optimize a data pipeline?"
```

- **Output**: Inference printed to terminal with max tokens = 100

---

## 💻 Environment Setup

```bash
git clone <repo-url>
cd interviewmate-finetune
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Make sure you're on a Mac with enough RAM (~16 GB for Falcon-1B).

---

## 📌 Notes

- PEFT model weights saved in `/left_model`
- Evaluation JSON file in `evaluation_data.json`
- Training script: `train.py` | Evaluation script: `run_evaluation.py`

---

**Maintained by Teja Chowdary • July 2025**
