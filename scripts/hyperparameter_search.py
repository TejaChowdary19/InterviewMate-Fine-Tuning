import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

MODEL_NAME = "tiiuae/falcon-rw-1b"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
dataset = load_dataset("json", data_files="data/ai_engineer_dataset.json", split="train")

def tokenize(example):
    return tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=256
    )

tokenized_dataset = dataset.map(tokenize)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define different hyperparameter configs
configs = [
    {"run_name": "run_A", "lr": 2e-4, "r": 8, "alpha": 16},
    {"run_name": "run_B", "lr": 1e-4, "r": 4, "alpha": 8},
    {"run_name": "run_C", "lr": 5e-5, "r": 2, "alpha": 4},
]

for cfg in configs:
    print(f"\n===== Starting {cfg['run_name']} =====\n")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.to("cpu")  # use CPU to avoid MPS issues

    training_args = TrainingArguments(
        output_dir=f"./results/{cfg['run_name']}",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        save_steps=10,
        save_total_limit=1,
        logging_steps=5,
        learning_rate=cfg["lr"],
        fp16=False,
        bf16=False,
        report_to="none",
        logging_dir=f"./results/{cfg['run_name']}/logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(f"./results/{cfg['run_name']}/peft_model")