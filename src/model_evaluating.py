import os
import sys
import json
import torch
import argparse
import io
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate

# Fix lỗi hiển thị ký tự đặc biệt trên Windows Terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_inference(model, tokenizer, samples):
    """Hàm bổ trợ để chạy dự đoán hàng loạt"""
    predictions = []
    references = []
    
    for item in tqdm(samples, desc="Dự đoán", leave=False):
        full_text = item['text']
        parts = full_text.split("<|im_start|>assistant\n")
        prompt = parts[0] + "<|im_start|>assistant\n"
        reference = parts[1].replace("<|im_end|>", "").strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        gen_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(gen_text)
        references.append(reference)
    
    return predictions, references

def evaluate_and_compare(val_file, output_metrics, model_id, adapter_path):
    # 1. Cấu hình Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Đọc dữ liệu validation
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    samples = val_data[:10]  # Lấy 10 mẫu để so sánh
    
    rouge = evaluate.load("rouge")
    final_results = {}

    # --- PHẦN 1: ĐÁNH GIÁ BASE MODEL ---
    print(f"\n--- [1/2] Dang danh gia BASE MODEL ({model_id}) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    base_preds, references = run_inference(base_model, tokenizer, samples)
    final_results["base_model"] = rouge.compute(predictions=base_preds, references=references)
    
    # Giải phóng VRAM của Base Model để nạp Adapter
    del base_model
    torch.cuda.empty_cache()

    # --- PHẦN 2: ĐÁNH GIÁ FINE-TUNED MODEL ---
    print(f"\n--- [2/2] Dang danh gia FINE-TUNED MODEL (Adapter) ---")
    # Load lại base để trộn adapter
    base_model_for_ft = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model_for_ft, adapter_path)
    ft_preds, _ = run_inference(ft_model, tokenizer, samples)
    final_results["finetuned_model"] = rouge.compute(predictions=ft_preds, references=references)

    # 3. Lưu kết quả so sánh
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    
    # 4. In bảng so sánh nhanh ra màn hình
    print("\n" + "="*50)
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Fine-tuned':<15}")
    print("-"*50)
    for m in ['rouge1', 'rouge2', 'rougeL']:
        base_v = final_results["base_model"][m]
        ft_v = final_results["finetuned_model"][m]
        print(f"{m:<15} | {base_v:<15.4f} | {ft_v:<15.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_metrics", type=str, default="metrics.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="models/qwen-medical-finetuned/final_adapter")
    args = parser.parse_args()
    
    evaluate_and_compare(args.val_file, args.output_metrics, args.model_id, args.adapter_path)