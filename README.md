# 🩺 Medical AI: Qwen2.5-0.5B Fine-Tuning & RAG Pipeline

[![DVC](https://img.shields.io/badge/MLOps-DVC-red.svg)](https://dvc.org/)
[![Model](https://img.shields.io/badge/LLM-Qwen2.5--0.5B-blue)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Dự án tập trung vào việc tinh chỉnh (Fine-tuning) mô hình ngôn ngữ lớn **Qwen2.5-0.5B** bằng kỹ thuật **QLoRA** để tối ưu hóa khả năng trả lời câu hỏi trong lĩnh vực y tế, kết hợp với hệ thống **RAG (Retrieval-Augmented Generation)** để đảm bảo tính chính xác dựa trên dữ liệu thực tế.



## 🌟 Key Features
- **QLoRA Fine-tuning**: Huấn luyện 4-bit giúp tối ưu tài nguyên (chỉ 0.28% tham số có thể huấn luyện), phù hợp với cấu hình máy hạn chế (8GB RAM).
- **DVC Pipeline**: Quản lý vòng đời dữ liệu và mô hình chuyên nghiệp, đảm bảo khả năng tái lập (reproducibility) 100%.
- **Comprehensive Evaluation**: Hệ thống đánh giá đối chứng trực tiếp giữa **Base Model** và **Fine-tuned Model** bằng chỉ số ROUGE.
- **RAG Integration**: Sử dụng **ChromaDB** làm Vector Database để truy xuất thông tin y khoa chính xác.
- **User Interface**: Giao diện Chatbot trực quan xây dựng bằng **Gradio**.

## 📊 Performance Evaluation
Kết quả đánh giá trên tập Validation cho thấy sự tăng trưởng vượt bậc sau khi Fine-tuning:

| Metric | Base Model | Fine-tuned Model | **Improvement** |
| :--- | :---: | :---: | :---: |
| **ROUGE-1** | 0.1058 | 0.1505 | **+42.2%** |
| **ROUGE-2** | 0.0074 | 0.0186 | **+151.3%** |
| **ROUGE-L** | 0.0707 | 0.1044 | **+47.6%** |

> **Note:** Chỉ số **ROUGE-2 tăng hơn 150%** chứng minh mô hình đã học được cách kết hợp các cụm từ chuyên môn y tế (Medical Bigrams) chính xác hơn hẳn mô hình gốc.



## 🛠️ Tech Stack
- **Core LLM**: `transformers`, `peft`, `bitsandbytes`, `accelerate`
- **Data Engineering**: `DVC`, `pandas`, `jsonlines`
- **Evaluation**: `evaluate`, `rouge-score`
- **Vector Store**: `ChromaDB`
- **UI Framework**: `Gradio`

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/psinguyenz/LLM-medical-finetune.git](https://github.com/psinguyenz/LLM-medical-finetune.git)
cd LLM-medical-finetune
pip install -r requirements.txt
```

###2. Reproduce Pipeline
Sử dụng DVC để chạy lại toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện:

```bash
dvc repro
```

###3. Kiểm tra kết quả đánh giá
Lệnh này sẽ hiển thị bảng so sánh các chỉ số đạt được:
📂 Cấu trúc dự án

```bash
python src/model_evaluating.py --output_metrics metrics.json

├── .dvc/                # Cấu hình quản lý dữ liệu phiên bản của DVC
├── data/                # Chứa dữ liệu thô và dữ liệu đã xử lý (DVC tracked)
├── src/                 # Mã nguồn chính xử lý LLM
│   ├── data_ingestion.py   # Nhập dữ liệu
│   ├── data_preprocessing.py  # Tiền xử lý dữ liệu sang format ChatML
│   ├── data_processing.py  # Data Collating
│   ├── model_building.py   # Script thực hiện Fine-tuning QLoRA
│   └── model_evaluating.py # Đánh giá đối chứng Base vs FT Model
├── .dvcignore           # Các file không cần DVC theo dõi
├── .gitignore           # Các file không cần Git theo dõi
├── dvc.lock             # Trạng thái hiện tại của pipeline (máy học đã chạy xong)
├── dvc.yaml             # Định nghĩa các stage huấn luyện và đánh giá
├── evaluation_details.json # Chi tiết kết quả dự đoán (output của model_evaluating)
├── metrics.json         # Tổng hợp chỉ số ROUGE (output của model_evaluating)
└── requirements.txt     # Danh sách thư viện cần thiết để chạy dự án
```

### P.S: use 

```bash
conda create -n llmmedical python=3.11 -y
conda activate llmmedical
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 # to use GPU
```

