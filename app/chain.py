# RAG链（Prompt + LLM）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """你是一个专业的知识库助手。请严格根据以下上下文回答问题。
如果上下文不包含答案，请直接说“根据已有资料，我无法回答这个问题”，不要编造信息。
答案中请标注信息来源。"""

def build_rag_chain(retriever, model_name: str = "gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "上下文：\n{context}\n\n问题：{question}")
    ])

    # 这里换成你 .env 里配置的对话模型，例如 "deepseek-chat"
    llm = ChatOpenAI(model="deepseek-chat", temperature=0)

    def format_docs(docs):
        return "\n\n".join(
            f"[来源: {d.metadata.get('source', '未知')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain