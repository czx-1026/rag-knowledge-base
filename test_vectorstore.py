from dotenv import load_dotenv
load_dotenv()

from app.vectorstore import load_vectorstore
from app.chain import build_rag_chain

print("正在加载本地向量库...")
vectorstore = load_vectorstore("./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("正在组装 RAG 链...")
chain = build_rag_chain(retriever, model_name="deepseek-chat") # 这里换成你用的模型

print("\n--- 开始问答测试 ---")
question = "len()函数是什么"
answer = chain.invoke(question)
print(f"\n问题：{question}")
print(f"回答：{answer}")