import os
import random
import pandas as pd
from datasets import load_dataset
import argparse

def ingest_data(dataset_name="ruslanmv/ai-medical-chatbot", output_path="data/raw"):
    """
    Tải dữ liệu từ Hugging Face và lưu dưới dạng CSV để DVC quản lý.
    """
    print(f"🚀 Đang bắt đầu tải dataset: {dataset_name}...")
    
    # 1. Load dataset
    dataset = load_dataset(dataset_name)
    data_split = dataset['train']
    
    # 2. Khám phá nhanh (Log ra console)
    num_samples = len(data_split)
    print(f"✅ Số lượng samples: {num_samples}")
    print(f"✅ Các cột hiện có: {data_split.column_names}")
    
    # 3. Chuyển sang Pandas DataFrame để dễ xử lý và lưu trữ
    df = pd.DataFrame(data_split)
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_path, exist_ok=True)
    
    # 4. Lưu dữ liệu
    file_name = "medical_chatbot_raw.csv"
    full_path = os.path.join(output_path, file_name)
    df.to_csv(full_path, index=False)
    
    print(f"✅ Dữ liệu đã được lưu tại: {full_path}")
    print("-" * 40)
    
    # In thử 3 ví dụ ngẫu nhiên để kiểm tra
    print("👀 Xem thử 3 dòng dữ liệu ngẫu nhiên:")
    print(df.sample(3))

if __name__ == "__main__":
    # Sử dụng argparse để bạn có thể thay đổi đường dẫn từ dòng lệnh nếu cần
    parser = argparse.ArgumentParser(description="Data Ingestion cho MLOps Pipeline")
    parser.add_argument("--output", type=str, default="data/raw", help="Thư mục lưu trữ dữ liệu")
    args = parser.parse_args()
    
    ingest_data(output_path=args.output)