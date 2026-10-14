import sys
import os

# 将当前文件的上一级目录（即项目根目录）加入到 Python 的搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 下面再接你原来的代码：
from app.vectorstore import load_vectorstore
from app.chain import build_rag_chain
from dotenv import load_dotenv

load_dotenv()
import json
from app.vectorstore import load_vectorstore
from app.chain import build_rag_chain
from dotenv import load_dotenv

load_dotenv()

def run_evaluation():
    # 1. 加载评测集
    with open("tests/eval_dataset.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # 2. 初始化你的 RAG 系统
    vectorstore = load_vectorstore("./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = build_rag_chain(retriever, model_name="deepseek-chat")

    total = len(test_cases)
    keyword_hits = 0
    correct_answers = 0

    print(f"开始评测，共 {total} 道题...\n")

    # 3. 循环测试
    for i, case in enumerate(test_cases):
        q = case["question"]
        keywords = case["relevant_keywords"]
        
        # 获取回答
        answer = chain.invoke(q)
        
        # 4. 简单粗暴的打分逻辑：检查关键词是否在回答里
        hit_count = sum(1 for kw in keywords if kw in answer)
        if hit_count >= len(keywords) * 0.5: # 命中一半以上关键词，算回答正确
            correct_answers += 1
            print(f"[{i+1}/{total}] ✅ 通过: {q}")
        else:
            print(f"[{i+1}/{total}] ❌ 未通过: {q}")
            print(f"    - 期望关键词: {keywords}")
            print(f"    - 实际回答: {answer[:100]}...\n")

    # 5. 输出最终报告
    accuracy = (correct_answers / total) * 100
    print("="*30)
    print(f"📊 评测报告")
    print(f"总问题数: {total}")
    print(f"回答正确数: {correct_answers}")
    print(f"🔥 准确率: {accuracy:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()