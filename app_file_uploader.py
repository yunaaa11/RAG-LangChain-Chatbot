#知识库自动化构建：文件上传与解析：通过 Streamlit 界面接收 .txt 文件
import streamlit as st
from knowledge_base import KnowledgeBaseSerivce
import time
#file_uploader
uploader_file=st.file_uploader(
    "请上传txt文件",
    type=['txt'],
    accept_multiple_files=False #只接受一个文件上传
)

#session_state是一个字典
if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseSerivce()
#如果不存在，说明是第一次运行该页面，需要创建一个 KnowledgeBaseSerivce 实例并存入会话状态。
if uploader_file is not None:
    #提取文件信息
    file_name=uploader_file.name
    file_type=uploader_file.type
    file_size=uploader_file.size/1024 #kb
    st.subheader(f"文件名:{file_name}")
    st.write(f"格式:{file_type}|大小:{file_size:.2f}KB")
    #解析内容
    text=uploader_file.getvalue().decode("utf-8")
    with st.spinner("载入知识库中..."):#转圈动画
        time.sleep(1)
        result=st.session_state["service"].upload_by_str(text,file_name)
        #从会话状态中取出之前初始化的服务实例。
        st.write(result)
        # st.write(text)