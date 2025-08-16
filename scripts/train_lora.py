import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

# Set your model
MODEL_NAME = "tiiuae/falcon-rw-1b"  # use smaller model to fit Mac memory

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Load dataset
dataset = load_dataset("json", data_files="data/ai_engineer_dataset.json", split="train")

# Tokenize dataset
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize)

# Load model and prepare for LoRA
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],  # Adjust based on model
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

# Send to CPU (MPS may crash on memory)
device = torch.device("cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora-falcon-ai-engineer",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    gradient_accumulation_steps=4,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=False,  # Disable fp16 (MPS not fully supported)
    bf16=False,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save the fine-tuned LoRA adapter
model.save_pretrained(f"left_model/run_{run_id}")
print(f"Saved model for run_{run_id}")
