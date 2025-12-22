import os
import sys
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer

# Fix lỗi hiển thị cho Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Class này sẽ được import vào src/train.py sau này.
    Nó giúp chỉ tính Loss trên phần trả lời của Assistant.
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        
        # Pad input_ids
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
            padding=True,
            max_length=self.tokenizer.model_max_length
        )
        
        labels = batch['input_ids'].clone()
        
        # Tìm Token ID của phần assistant để mask
        assistant_start_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        assistant_start_id = assistant_start_ids[-1] if assistant_start_ids else None

        for i, sample_input_ids in enumerate(input_ids):
            if assistant_start_id in sample_input_ids:
                idx = sample_input_ids.index(assistant_start_id)
                # Mask toàn bộ phần trước Assistant (User & System) bằng -100
                labels[i, :idx + 1] = -100
            else:
                labels[i, :] = -100
        
        batch["labels"] = labels
        return batch

def run_tokenization(input_dir, output_dir, model_id):
    print(f"--- Dang Tokenize du lieu voi model: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Thiết lập pad_token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dữ liệu .jsonl từ bước Preprocessing
    train_file = os.path.join(input_dir, "train.jsonl")
    val_file = os.path.join(input_dir, "val.jsonl")
    
    dataset = load_dataset("json", data_files={"train": train_file, "val": val_file})

    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=1024,
            padding=False 
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Chạy Tokenization
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )

    # Lưu xuống ổ cứng dạng thư mục dataset
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset["train"].save_to_disk(os.path.join(output_dir, "train"))
    tokenized_dataset["val"].save_to_disk(os.path.join(output_dir, "val"))
    print(f"✅ Da luu Tokenized dataset tai: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="data/tokenized")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    
    run_tokenization(args.input_dir, args.output_dir, args.model_id)