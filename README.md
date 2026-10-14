# 🤖 基于 LangChain 与 Chroma 的本地 RAG 知识库问答助手

本项目是一个基于大语言模型（LLM）的检索增强生成（RAG）应用。它能够读取本地指定目录下的 PDF 文档，通过本地向量化模型进行嵌入与存储，并基于检索到的上下文，调用大模型（DeepSeek）进行精准问答，同时支持来源溯源。

## 📸 效果展示
<img width="2862" height="1533" alt="demo png" src="https://github.com/user-attachments/assets/d227236d-cc9d-4848-b44b-26dc16d63cf0" />



## ✨ 核心功能
- **多文档加载**：支持批量加载并解析本地 PDF 文档。
- **智能切分**：采用 `RecursiveCharacterTextSplitter` 针对中文语境进行文本切分。
- **本地向量化**：使用 HuggingFace 的 `BAAI/bge-small-zh-v1.5` 模型进行本地 Embedding，无需调用昂贵的云端 API。
- **向量存储与检索**：使用 Chroma 向量数据库持久化存储，支持相似度 Top-K 检索。
- **大模型问答**：接入 DeepSeek API（兼容 OpenAI 协议），基于检索上下文生成回答。
- **来源溯源**：回答中自动标注参考的文档来源。
- **可视化界面**：基于 Streamlit 搭建的交互式聊天 Web 界面。

## 🛠️ 技术栈
- **核心框架**：LangChain (LCEL)
- **向量数据库**：Chroma
- **嵌入模型**：HuggingFace `BAAI/bge-small-zh-v1.5`
- **大语言模型**：DeepSeek-Chat
- **前端界面**：Streamlit
- **运行环境**：Python 3.12 + venv

## 🔄 核心流程
```text
用户提问 
  -> 本地 Embedding 模型将问题向量化 
  -> Chroma 向量库进行相似度检索 (Top-K) 
  -> 拼接检索到的上下文与 Prompt 
  -> 调用 DeepSeek 大模型生成回答 
  -> 返回带有来源标注的答案
