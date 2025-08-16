# scripts/evaluate.py

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_DIR = "models/gpt2-finetuned"

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model.eval()

# Load test set
dataset = load_dataset("csv", data_files={"test": "data/test.csv"})["test"]

results = []
smooth_fn = SmoothingFunction().method1

print("📊 Evaluating on test set...\n")

for item in dataset:
    question = item["text"].split("Question:")[-1].split("Answer:")[0].strip()
    expected = item["text"].split("Answer:")[-1].strip()

    inputs = tokenizer.encode(question + tokenizer.eos_token, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted = decoded[len(question):].strip()

    bleu = sentence_bleu([expected.split()], predicted.split(), smoothing_function=smooth_fn)
    exact_match = int(predicted.strip().lower() == expected.strip().lower())

    results.append({
        "question": question,
        "expected": expected,
        "predicted": predicted,
        "bleu_score": bleu,
        "exact_match": exact_match
    })

# Save to CSV
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/eval_results.csv", index=False)

print("✅ Evaluation Complete:")
print(f"🔵 Avg BLEU Score: {df['bleu_score'].mean():.2f}")
print(f"✅ Exact Match Accuracy: {df['exact_match'].mean() * 100:.1f}%")
print("📁 Saved detailed results to results/eval_results.csv")
