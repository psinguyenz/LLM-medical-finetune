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

# Import hàm create RAG system
from model_RAG import create_rag_system

# Fix lỗi hiển thị ký tự đặc biệt trên Windows Terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

SYSTEM_PROMPT = "You are a professional and detailed medical assistant, providing information based on scientific evidence."

def extract_assistant_response(output_ids, inputs, tokenizer):
    """Extract the clean assistant response from the model's output IDs."""
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    assistant_prefix = "<|im_start|>assistant\n"
    
    start_index = generated_text.find(assistant_prefix)
    if start_index != -1:
        response_with_end_tokens = generated_text[start_index + len(assistant_prefix):]
        response = response_with_end_tokens.split("<|im_end|>")[0].strip()
    else:
        input_len = inputs['input_ids'].shape[1]
        response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
    return response

def generate_response_base_finetuned(model, tokenizer, question, max_new_tokens=256):
    """Generate response từ base model hoặc fine-tuned model"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return extract_assistant_response(output_ids, inputs, tokenizer)

def run_inference_base_finetuned(model, tokenizer, samples):
    """Hàm bổ trợ để chạy dự đoán hàng loạt cho base/finetuned model"""
    predictions = []
    references = []
    
    for item in tqdm(samples, desc="Dự đoán", leave=False):
        full_text = item['text']
        parts = full_text.split("<|im_start|>assistant\n")
        if len(parts) < 2:
            continue
        prompt = parts[0] + "<|im_start|>assistant\n"
        reference = parts[1].replace("<|im_end|>", "").strip()
        
        # Extract question từ prompt
        user_part = prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
        
        pred = generate_response_base_finetuned(model, tokenizer, user_part)
        predictions.append(pred)
        references.append(reference)
    
    return predictions, references

def run_inference_rag(chain, samples):
    """Hàm bổ trợ để chạy dự đoán hàng loạt cho RAG model"""
    predictions = []
    references = []
    
    for item in tqdm(samples, desc="Dự đoán RAG", leave=False):
        full_text = item['text']
        parts = full_text.split("<|im_start|>assistant\n")
        if len(parts) < 2:
            continue
        reference = parts[1].replace("<|im_end|>", "").strip()
        
        # Extract question từ prompt
        user_part = full_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
        
        try:
            pred = chain.invoke(user_part)
            predictions.append(pred)
            references.append(reference)
        except Exception as e:
            print(f"Error in RAG inference: {e}")
            predictions.append("")
            references.append(reference)
    
    return predictions, references

def evaluate_and_compare(val_file, output_metrics, model_id, adapter_path, vectorstore_path):
    # Kiểm tra và đảm bảo sử dụng GPU
    print("\n" + "="*60)
    print("🔍 KIỂM TRA HỆ THỐNG")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"✅ Sử dụng GPU để tính toán")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA không khả dụng, sẽ sử dụng CPU")
    
    print("="*60 + "\n")
    
    # 1. Cấu hình Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Đọc dữ liệu validation
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    samples = val_data[:5]  # Lấy 50 mẫu để so sánh (có thể tăng lên nếu muốn)
    
    rouge = evaluate.load("rouge")
    final_results = {}

    # --- PHẦN 1: ĐÁNH GIÁ BASE MODEL ---
    print(f"\n--- [1/3] Đang đánh giá BASE MODEL ({model_id}) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )
    base_preds, references = run_inference_base_finetuned(base_model, tokenizer, samples)
    final_results["base_model"] = rouge.compute(predictions=base_preds, references=references)
    
    # Giải phóng VRAM của Base Model
    del base_model
    torch.cuda.empty_cache()

    # --- PHẦN 2: ĐÁNH GIÁ FINE-TUNED MODEL ---
    print(f"\n--- [2/3] Đang đánh giá FINE-TUNED MODEL (Adapter) ---")
    base_model_for_ft = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )
    ft_model = PeftModel.from_pretrained(base_model_for_ft, adapter_path)
    ft_preds, _ = run_inference_base_finetuned(ft_model, tokenizer, samples)
    final_results["finetuned_model"] = rouge.compute(predictions=ft_preds, references=references)
    
    # Giải phóng VRAM của Fine-tuned Model
    del ft_model
    del base_model_for_ft
    torch.cuda.empty_cache()

    # --- PHẦN 3: ĐÁNH GIÁ FINE-TUNED + RAG MODEL ---
    print(f"\n--- [3/3] Đang đánh giá FINE-TUNED + RAG MODEL ---")
    # create_rag_system sẽ tự động load vectorstore nếu đã tồn tại
    rag_chain, _ = create_rag_system(
        model_base_id=model_id,
        adapter_path=adapter_path,
        dataset_name="ruslanmv/ai-medical-chatbot",  # Cần để tạo mới nếu chưa có, nhưng sẽ không dùng nếu đã có vectorstore
        vectorstore_path=vectorstore_path
    )
    rag_preds, _ = run_inference_rag(rag_chain, samples)
    final_results["finetuned_rag_model"] = rouge.compute(predictions=rag_preds, references=references)

    # 3. Lưu kết quả so sánh
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    # 4. In bảng so sánh nhanh ra màn hình
    print("\n" + "="*70)
    print(f"{'Metric':<15} | {'Base Model':<15} | {'Fine-tuned':<15} | {'Fine-tuned+RAG':<15}")
    print("-"*70)
    for m in ['rouge1', 'rouge2', 'rougeL']:
        base_v = final_results["base_model"][m]
        ft_v = final_results["finetuned_model"][m]
        rag_v = final_results["finetuned_rag_model"][m]
        print(f"{m:<15} | {base_v:<15.4f} | {ft_v:<15.4f} | {rag_v:<15.4f}")
    print("="*70)
    
    print(f"\n✅ Kết quả đã được lưu tại: {output_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_metrics", type=str, default="metrics.json")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="models/qwen-medical-finetuned/final_adapter")
    parser.add_argument("--vectorstore_path", type=str, default="data/rag_vectorstore")
    args = parser.parse_args()
    
    evaluate_and_compare(args.val_file, args.output_metrics, args.model_id, args.adapter_path, args.vectorstore_path)
