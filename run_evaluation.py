import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# === Configuration ===
BASE_MODEL_NAME = "tiiuae/falcon-7b"  # <-- make sure this matches what you used for training
PEFT_MODEL_DIR = "./lora-falcon-ai-engineer/checkpoint-1875"
EVAL_FILE = "evaluation_data.json"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, PEFT_MODEL_DIR)
model.eval()

# === Load evaluation dataset ===
with open(EVAL_FILE, "r") as f:
    eval_data = json.load(f)

# === Load ROUGE metric ===
rouge = evaluate.load("rouge")

# === Run evaluation ===
all_preds = []
all_refs = []

for i, item in enumerate(eval_data):
    prompt = item["input"]
    expected = item["expected"]

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    all_preds.append(decoded_output)
    all_refs.append(expected.strip().lower())

    print(f"\n=== Prompt {i+1} ===")
    print("Prompt:", prompt)
    print("Response:", decoded_output)
    print("Expected:", expected)

# === Compute ROUGE ===
results = rouge.compute(predictions=all_preds, references=all_refs)
print("\n=== Evaluation Results ===")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
