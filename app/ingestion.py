# 文档加载与切分
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split(data_dir: str = "./data"):
    """加载目录下所有PDF并切分"""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    raw_docs = loader.load()
    print(f"加载了 {len(raw_docs)} 个文档")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"切分为 {len(chunks)} 个文本块")
    return chunks