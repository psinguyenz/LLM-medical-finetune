import os
import sys
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Import Class DataCollator từ bước processing
from data_processing import DataCollatorForCompletionOnlyLM

# Fix encoding cho Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def build_and_train(input_dir, output_dir, model_id):
    print(f"--- Đang khởi tạo mô hình: {model_id} ---")
    
    # 1. Cấu hình Quantization (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )

    # 3. Chuẩn bị model cho QLoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Thiết lập cấu hình LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load dữ liệu đã tokenize
    print(f"--- Đang nạp dữ liệu từ: {input_dir} ---")
    tokenized_train = load_from_disk(os.path.join(input_dir, "train"))
    tokenized_val = load_from_disk(os.path.join(input_dir, "val"))

    # 6. Khởi tạo Data Collator
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

    # 7. Thiết lập tham số huấn luyện (Chế độ chạy thử 0.1 epoch)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,                # Giảm xuống 1 epoch như bạn muốn
        per_device_train_batch_size=1,     # Bắt buộc là 1
        gradient_accumulation_steps=32,    # Tăng lên 32 để ít phải ghi dữ liệu vào RAM hơn
        learning_rate=2e-4,
        logging_steps=5,                   # Giảm tần suất in log
        eval_strategy="no",                # Tắt hoàn toàn để tiết kiệm RAM/VRAM
        save_strategy="no",                # Không lưu checkpoint giữa chừng, chỉ lưu cái cuối
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,          # Không dùng tiến trình phụ
        report_to="none"
    )

    # 8. Chạy Trainer (Đã bỏ callbacks)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )

    print("🚀 Bắt đầu quá trình huấn luyện...")
    trainer.train()

    # 9. Lưu Adapter
    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    print(f"✅ Huấn luyện hoàn tất! Adapter được lưu tại: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/tokenized")
    parser.add_argument("--output_dir", type=str, default="models/qwen-medical-finetuned")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    
    build_and_train(args.input_dir, args.output_dir, args.model_id)