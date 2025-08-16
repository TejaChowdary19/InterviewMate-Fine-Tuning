from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-v0.1"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("🔄 Loading model (full precision, no quantization)...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print("✅ Model and tokenizer loaded successfully!")
