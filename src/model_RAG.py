import os
import sys
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Fix encoding cho Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def create_rag_system(model_base_id, adapter_path, dataset_name, vectorstore_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=50, k_retrieval=3):
    """
    Tạo hệ thống RAG với fine-tuned model, embeddings và vectorstore.
    Nếu vectorstore đã tồn tại, sẽ load lại thay vì tạo mới.
    
    Args:
        model_base_id: ID của base model (ví dụ: "Qwen/Qwen2.5-1.5B-Instruct")
        adapter_path: Đường dẫn tới adapter đã fine-tuned
        dataset_name: Tên dataset trên Hugging Face (ví dụ: "ruslanmv/ai-medical-chatbot")
        vectorstore_path: Đường dẫn lưu/load vectorstore
        embedding_model_name: Tên model embedding
        chunk_size: Kích thước chunk
        chunk_overlap: Độ trùng lặp giữa các chunks
        k_retrieval: Số lượng documents để retrieve
    
    Returns:
        chain: RAG chain
        vectorstore: FAISS vectorstore
    """
    print("--- Đang khởi tạo hệ thống RAG ---")
    
    # Kiểm tra GPU
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
    
    # 1. Load fine-tuned model và tạo pipeline
    print(f"--- Đang tải base model: {model_base_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_base_id, trust_remote_code=True)
    
    # Cấu hình 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_base_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )
    
    print(f"--- Đang tải adapter từ: {adapter_path} ---")
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Tạo Hugging Face Pipeline
    print("--- Đang tạo pipeline ---")
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=finetuned_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Bọc pipeline thành LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ LLM Fine-Tuned đã được khởi tạo.")
    
    # 2. Load hoặc tạo vectorstore
    embedding_model_name = embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2"
    print(f"--- Đang tải embedding model: {embedding_model_name} ---")
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
    print(f"✅ Đã tải embedding model.")
    
    # Kiểm tra xem vectorstore đã tồn tại chưa
    index_path = os.path.join(vectorstore_path, "index.faiss")
    if os.path.exists(index_path):
        print(f"--- Vectorstore đã tồn tại, đang load từ: {vectorstore_path} ---")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print(f"✅ Vectorstore đã được load thành công.")
    else:
        print(f"--- Vectorstore chưa tồn tại, đang tạo mới ---")
        # Load dataset và tạo documents
        print(f"--- Đang tải dataset: {dataset_name} ---")
        subset_size = 1000
        dataset = load_dataset(dataset_name, split=f'train[:{subset_size}]')
        
        rag_documents = []
        for sample in dataset:
            content = (
                f"Câu hỏi bệnh nhân: {sample['Patient']}\n"
                f"Tình trạng/Mô tả: {sample['Description']}\n"
                f"Câu trả lời chuyên môn: {sample['Doctor']}"
            )
            metadata = {"source": "ai-medical-chatbot-dataset"}
            rag_documents.append(Document(page_content=content, metadata=metadata))
        
        print(f"✅ Đã tải {len(rag_documents)} mẫu từ dataset.")
        
        # Chia nhỏ documents (Text Splitting)
        print("--- Đang chia nhỏ documents ---")
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks_of_text = text_splitter.split_documents(rag_documents)
        print(f"✅ Đã chia thành {len(chunks_of_text)} chunks.")
        
        # Tạo vectorstore
        print("--- Đang tạo vectorstore và indexing ---")
        vectorstore = FAISS.from_documents(chunks_of_text, embeddings)
        
        # Lưu vectorstore
        os.makedirs(os.path.dirname(vectorstore_path) if os.path.dirname(vectorstore_path) else ".", exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        print(f"✅ Vectorstore đã được lưu tại: {vectorstore_path}")
    
    # 3. Khởi tạo Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
    print(f"✅ Retriever đã được khởi tạo (k={k_retrieval}).")
    
    # 4. Tạo prompt template
    template = """Answer the question based on the following context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Tạo RAG chain
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ Hệ thống RAG đã được khởi tạo hoàn tất!")
    
    return chain, vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thiết lập hệ thống RAG với fine-tuned model")
    parser.add_argument("--model_base_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", 
                        help="ID của base model")
    parser.add_argument("--adapter_path", type=str, default="models/qwen-medical-finetuned/final_adapter",
                        help="Đường dẫn tới adapter đã fine-tuned")
    parser.add_argument("--dataset_name", type=str, default="ruslanmv/ai-medical-chatbot",
                        help="Tên dataset trên Hugging Face")
    parser.add_argument("--vectorstore_path", type=str, default="data/rag_vectorstore",
                        help="Đường dẫn lưu vectorstore")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Tên model embedding")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Kích thước chunk")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Độ trùng lặp giữa các chunks")
    parser.add_argument("--k_retrieval", type=int, default=3,
                        help="Số lượng documents để retrieve")
    
    args = parser.parse_args()
    
    chain, vectorstore = create_rag_system(
        model_base_id=args.model_base_id,
        adapter_path=args.adapter_path,
        dataset_name=args.dataset_name,
        vectorstore_path=args.vectorstore_path,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k_retrieval=args.k_retrieval
    )
    
    # Test query
    print("\n--- TEST QUERY ---")
    query = "Tôi bị tê và ngứa ran ở tay vào ban đêm, nguyên nhân là gì và cách điều trị thông thường là gì?"
    print(f"Query: {query}\n")
    response = chain.invoke(query)
    print(f"Response: {response}")

