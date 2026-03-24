🤖 基于 LangChain + Qwen 的 RAG 智能客服系统

本项目是一个基于 Retrieval-Augmented Generation (RAG) 技术的智能客服原型。它能够读取本地 TXT 或 PDF 格式的知识库，通过向量化存储与检索，实现基于特定私有知识的精准问答。

🌟 核心功能
1.多格式知识导入：支持 .txt 和 .pdf 文件上传，自动提取文本。
2.分段 MD5 增量更新：对文件进行颗粒度拆分，通过 MD5 校验仅存储“新内容”，极大节省向量化 API 开销。
3.双引擎检索优化：
-Top-K 检索：动态调整返回的知识片段数量。
-分数阈值过滤：自动剔除相关度较低的干扰信息，确保回复准确性。
4.流式对话体验：集成阿里通义千问（Qwen）大模型，支持打字机式的流式回复。
5.持久化存储：对话历史自动存入本地 JSON，重启不丢失上下文；向量数据持久化于本地 ChromaDB。

🛠️ 技术栈
-Frontend: Streamlit
-Orchestration: LangChain
-LLM & Embedding: Alibaba DashScope (Qwen-Max / Text-Embedding-v4)
-Vector Database: ChromaDB

🚀 快速开始
1. 克隆项目
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名
2. 安装依赖
pip install -r requirements.txt
3. 配置 API Key
在 config_data.py 中填入你的通义千问 API Key：
dashscope_api_key = "你的API_KEY"
4. 运行系统
首先运行上传界面，录入你的知识库（如：尺码推荐表）：
streamlit run app_file_uploader.py
然后启动对话主界面进行测试：
streamlit run app_qa.py

📂 目录结构说明
app_file_uploader.py: 知识库自动化构建界面。
app_qa.py: 智能客服对话前端。
knowledge_base.py: 核心逻辑类，负责文件解析、MD5 校验与入库。
rag.py: RAG 链条逻辑实现，处理检索与 Prompt 生成。
vector_stores.py: 封装向量数据库初始化与检索器配置。
config_data.py: 全局配置参数（模型名称、K值、路径等）。

📝 许可证
本项目采用 MIT License 开源。