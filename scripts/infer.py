from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_path = "./models/fine-tuned-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("💬 InterviewMate is ready. Type your interview question (type 'exit' to quit):\n")

while True:
    user_input = input("🧑 You: ")
    if user_input.lower() == "exit":
        print("👋 Exiting InterviewMate. Good luck with your prep!")
        break

    # Encode input and build attention mask
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,            # deterministic output
            num_beams=4,                # beam search
            early_stopping=True
        )

    # Decode and print output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Clean response (remove question if repeated)
    cleaned = response.replace(user_input.strip(), "").strip()
    print(f"🤖 InterviewMate: {cleaned}\n")
