import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate

def evaluate_and_compare(val_file, output_metrics, model_id, adapter_path):
    # 1. Cấu hình Load mô hình tiết kiệm RAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. Load Fine-tuned Model (Adapter)
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # 4. Chuẩn bị dữ liệu và chỉ số
    rouge = evaluate.load("rouge")
    
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    samples = val_data[:10]  # Lấy 10 câu đầu để đánh giá
    
    results_to_save = []
    all_ft_preds = []
    all_references = []

    print(f"🚀 Đang đánh giá trên {len(samples)} mẫu từ tập Validation...")

    for item in tqdm(samples):
        full_text = item['text']
        # Tách prompt và reference từ format ChatML
        parts = full_text.split("<|im_start|>assistant\n")
        prompt = parts[0] + "<|im_start|>assistant\n"
        reference = parts[1].replace("<|im_end|>", "").strip()

        # Sinh câu trả lời từ Fine-tuned model
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = ft_model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        ft_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        all_ft_preds.append(ft_response)
        all_references.append(reference)

        results_to_save.append({
            "prompt": prompt,
            "reference": reference,
            "finetuned_response": ft_response
        })

    # 5. Tính toán ROUGE
    rouge_results = rouge.compute(predictions=all_ft_preds, references=all_references)

    # 6. Lưu file JSON kết quả chi tiết và Metrics
    # Lưu metrics cho DVC
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(rouge_results, f, indent=4)
        
    # Lưu log chi tiết để bạn xem lại câu trả lời
    with open("evaluation_details.json", "w", encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Hoàn tất! Chỉ số ROUGE: {rouge_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_metrics", type=str, default="metrics.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="models/qwen-medical-finetuned/final_adapter")
    args = parser.parse_args()
    
    evaluate_and_compare(args.val_file, args.output_metrics, args.model_id, args.adapter_path)