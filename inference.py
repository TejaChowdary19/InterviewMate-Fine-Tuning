import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

# Paths
base_model_name = "tiiuae/falcon-rw-1b"
peft_model_dir = "lora-falcon-ai-engineer/checkpoint-1875"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))  # 🔧 CRUCIAL!

# Load LoRA adapter
model = PeftModel.from_pretrained(model, peft_model_dir, local_files_only=True)

# Inference
model.eval()
device = torch.device("cpu")
model.to(device)

prompt = "Give me tips to prepare for a data engineering interview."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Response ===\n", response)
