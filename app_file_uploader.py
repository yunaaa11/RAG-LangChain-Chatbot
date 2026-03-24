#知识库自动化构建：文件上传与解析：通过 Streamlit 界面接收 .txt 文件
import streamlit as st
from knowledge_base import KnowledgeBaseSerivce
import time
#file_uploader
uploader_file=st.file_uploader(
    "请上传知识库文件 (支持 TXT, PDF)",
    type=['txt', 'pdf'],
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
    
    with st.spinner("载入知识库中..."):
        # 判断文件类型并处理
        if file_name.lower().endswith('.pdf'):
            # 对于 PDF，我们直接传递 bytes 或保存临时文件
            # 建议将解析逻辑封装在 service.upload_by_file 中
            result = st.session_state["service"].upload_by_file(uploader_file, file_name)
        else:
           # TXT 逻辑：确认为文本时再读取
            try:
                # 确保指针在开头
                uploader_file.seek(0)
                content = uploader_file.read().decode("utf-8")
                result = st.session_state["service"].upload_by_str(content, file_name)
            except Exception as e:
                result = f"文本解析失败: {e}"
        st.write(result)