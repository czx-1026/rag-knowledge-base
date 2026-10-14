# 向量化与存储
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 强制使用国内镜像

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(chunks, persist_dir: str = "./chroma_db"):
    """将chunks向量化并存入Chroma（本地Embedding模型）"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ 向量库已构建，持久化到 {persist_dir}")
    return vectorstore

def load_vectorstore(persist_dir: str = "./chroma_db"):
    """从磁盘加载已有向量库"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print(f"✅ 已从 {persist_dir} 加载向量库")
    return vectorstore