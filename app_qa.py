#流式输出：前端 app_qa.py 使用 st.write_stream 实现了打字机效果，提升用户体验。
import streamlit as st
from rag import RagService
import config_data as config

st.title("智能客服")
st.divider()

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 渲染历史消息
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入
prompt = st.chat_input()

if prompt:
    # 1. 立即显示用户输入并存入历史
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 2. AI 思考与流式输出
    with st.spinner("AI思考中..."):
        # 获取流
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        
        # 使用 write_stream 并获取其返回值
        # Streamlit 的 write_stream 会自动处理生成器并返回完整的字符串
        with st.chat_message("assistant"):
            full_response = st.write_stream(res_stream)
        
        # 3. 将完整的字符串存入历史（注意角色是 assistant）
        st.session_state["messages"].append({"role": "assistant", "content": full_response})