from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

# Load the cleaned dataset
data_path = os.path.join("data", "interview_qa_cleaned.csv")
dataset = load_dataset("csv", data_files={"train": data_path})

# Tokenization function
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    result["labels"] = result["input_ids"].copy()  # ✅ Add this line
    return result

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# Split into train and validation
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

# Save processed dataset
dataset_dict.save_to_disk("data/tokenized_dataset")

print("✅ Tokenization and splitting complete! Saved to: data/tokenized_dataset")
