import streamlit as st
from dotenv import load_dotenv

# 同样，环境变量必须在最前面加载
load_dotenv()

from app.vectorstore import load_vectorstore
from app.chain import build_rag_chain

# 页面基础配置
st.set_page_config(page_title="我的 RAG 知识库", page_icon="🤖")
st.title("🤖 我的 RAG 知识库问答助手")
st.caption("基于 LangChain + Chroma + DeepSeek，处理了 330 份 PDF 文档")

# 【关键优化】使用缓存，避免每次提问都重新加载向量库
@st.cache_resource
def init_rag():
    print("正在初始化 RAG 系统...")
    vectorstore = load_vectorstore("./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # 这里记得换成你实际使用的模型名字
    chain = build_rag_chain(retriever, model_name="deepseek-chat")
    return chain

# 初始化链路（第一次运行会加载向量库，之后走缓存，秒开）
chain = init_rag()

# 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你想问的问题..."):
    # 显示用户气泡
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成助手气泡
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})