#!/usr/bin/env python3
"""
Data Preparation Script for InterviewMate Fine-tuning
Handles dataset splitting, cleaning, and formatting for fine-tuning
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import Dataset, DatasetDict
import pandas as pd

def load_and_clean_dataset(data_path: str) -> List[Dict[str, str]]:
    """Load and clean the AI engineer dataset"""
    print("📊 Loading dataset...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📈 Loaded {len(data)} samples")
    
    # Clean and validate data
    cleaned_data = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            text = item["text"].strip()
            if text and len(text) > 10:  # Filter out very short entries
                cleaned_data.append({"text": text})
    
    print(f"🧹 Cleaned dataset: {len(cleaned_data)} samples")
    return cleaned_data

def create_train_val_test_split(data: List[Dict[str, str]], 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               seed: int = 42) -> Tuple[List, List, List]:
    """Split data into train/validation/test sets"""
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_data)} samples ({len(train_data)/n*100:.1f}%)")
    print(f"   Validation: {len(val_data)} samples ({len(val_data)/n*100:.1f}%)")
    print(f"   Test: {len(test_data)} samples ({len(test_data)/n*100:.1f}%)")
    
    return train_data, val_data, test_data

def save_splits(train_data: List, val_data: List, test_data: List, 
                output_dir: str = "data"):
    """Save the split datasets"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSON
    with open(output_path / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "validation.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save as CSV for easier analysis
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "validation.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"💾 Saved splits to {output_dir}/")

def create_huggingface_dataset(train_data: List, val_data: List, 
                              test_data: List) -> DatasetDict:
    """Create HuggingFace DatasetDict for training"""
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print("🤗 Created HuggingFace DatasetDict")
    return dataset_dict

def main():
    """Main data preparation pipeline"""
    print("🚀 Starting InterviewMate Data Preparation...")
    
    # Load and clean dataset
    data_path = "data/ai_engineer_dataset.json"
    if not Path(data_path).exists():
        print(f"❌ Dataset not found at {data_path}")
        return
    
    data = load_and_clean_dataset(data_path)
    
    # Create splits
    train_data, val_data, test_data = create_train_val_test_split(data)
    
    # Save splits
    save_splits(train_data, val_data, test_data)
    
    # Create HuggingFace dataset
    dataset_dict = create_huggingface_dataset(train_data, val_data, test_data)
    
    # Save HuggingFace dataset
    dataset_dict.save_to_disk("data/tokenized_dataset")
    print("💾 Saved HuggingFace dataset to data/tokenized_dataset/")
    
    print("✅ Data preparation complete!")
    
    # Print sample statistics
    print("\n📊 Sample Statistics:")
    print(f"   Average text length (train): {sum(len(x['text']) for x in train_data) / len(train_data):.1f} chars")
    print(f"   Average text length (val): {sum(len(x['text']) for x in val_data) / len(val_data):.1f} chars")
    print(f"   Average text length (test): {sum(len(x['text']) for x in test_data) / len(test_data):.1f} chars")

if __name__ == "__main__":
    main()
