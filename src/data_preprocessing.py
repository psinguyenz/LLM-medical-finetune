import os
import pandas as pd
import argparse
import sys
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = "You are a professional and detailed medical assistant, providing information based on scientific evidence."

def format_instruction(sample):
    """Format sample theo chuẩn ChatML của Qwen"""
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{sample['Patient']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['Doctor']}<|im_end|>"
    )
    return {"text": text}

def preprocess_data(input_file, output_dir):
    print(f"--- Dang xu ly du lieu tu: {input_file} ---")
    
    # 1. Load data tu file CSV ma buoc Ingestion da tao ra
    df = pd.read_csv(input_file)
    raw_dataset = Dataset.from_pandas(df)
    
    # 2. Apply formatting bang ham .map (nhanh hon loop for)
    formatted_dataset = raw_dataset.map(
        format_instruction, 
        remove_columns=raw_dataset.column_names, # Xoa cac cot cu (Patient, Doctor, Description)
        desc="Formatting samples"
    )
    
    # 3. Shuffle va Limit
    TARGET_SAMPLES = 1000
    SEED = 42
    shuffled_dataset = formatted_dataset.shuffle(seed=SEED)
    
    if len(shuffled_dataset) > TARGET_SAMPLES:
        limited_dataset = shuffled_dataset.select(range(TARGET_SAMPLES))
    else:
        limited_dataset = shuffled_dataset
        
    # 4. Split Train/Validation (90/10)
    split_datasets = limited_dataset.train_test_split(test_size=0.1, seed=SEED)
    
    # 5. Luu du lieu xuong o cung
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    
    split_datasets['train'].to_json(train_path, orient="records", lines=True)
    split_datasets['test'].to_json(val_path, orient="records", lines=True)
    
    print(f"✅ Preprocessing hoan tat!")
    print(f" - Train: {len(split_datasets['train'])} mau -> {train_path}")
    print(f" - Val: {len(split_datasets['test'])} mau -> {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/medical_chatbot_raw.csv")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)