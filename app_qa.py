#流式输出：前端 app_qa.py 使用 st.write_stream 实现了打字机效果，提升用户体验。
import streamlit as st
from rag import RagService
import config_data as config
# 统一导入，删除对 knowledge_base 的引用
from vector_stores import VectorStoreService 
from langchain_community.embeddings import DashScopeEmbeddings

st.title("智能客服")
st.divider()

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

# 1. 初始化知识库服务并获取增强检索器
if "enhanced_retriever" not in st.session_state:
    with st.spinner("正在加载工业级检索引擎..."):
        # 初始化向量服务
        kb_service = VectorStoreService(DashScopeEmbeddings(
            model=config.embedding_model_name,
            dashscope_api_key=config.dashscope_api_key
        ))
        # 构建并存入 session_state
        st.session_state["enhanced_retriever"] = kb_service.get_retriever()

# 2. 初始化 RagService，并将增强后的检索器传入
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService(retriever=st.session_state["enhanced_retriever"])

# 渲染历史消息
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 2. AI 思考与流式输出
    with st.spinner("AI思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        
        with st.chat_message("assistant"):
            full_response = st.write_stream(res_stream)
        
        st.session_state["messages"].append({"role": "assistant", "content": full_response})