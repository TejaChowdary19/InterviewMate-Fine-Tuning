import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the cleaned dataset
df = pd.read_csv("data/interview_qa_cleaned.csv")

# Basic cleanup if needed
df = df.dropna()
df = df[df["text"].str.contains("Question:") & df["text"].str.contains("Answer:")]

# Split 85% train, 15% test
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

# Create output directory if not exists
os.makedirs("data", exist_ok=True)

# Save splits
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Split complete: train.csv and test.csv saved in data/")
