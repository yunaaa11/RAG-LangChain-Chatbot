# 🤖 基于 LangChain 的 RAG 智能客服系统

本项目是一个基于 **Retrieval-Augmented Generation (RAG)** 技术的智能客服原型。它能够读取本地 TXT 或 PDF 格式的知识库，通过向量化存储与检索，实现基于特定私有知识的精准问答。

---

## 🌟 核心功能

* **工业级混合检索 (Hybrid RAG)**：结合 ChromaDB 语义向量检索与 BM25 关键词检索，通过 EnsembleRetriever 实现多路召回，大幅提升在专业术语和长文本下的检索精度。

* **重排序优化 (Reranking)**：集成 Flashrank 精排引擎（ms-marco-TinyBERT 模型），对召回片段进行二次筛选，有效解决大模型“迷失在中间”的问题。

* **智能增量更新**：支持 .txt / .pdf 多格式导入，采用 MD5 颗粒度分段校验，仅对更新内容进行向量化，显著降低 API 调用成本。

* **极致对话体验**：基于 Streamlit 与 阿里通义千问 (Qwen)，实现全链路流式打字机输出，支持本地持久化的多轮对话上下文记忆。

* **工程化日志审计**：内置自动化日志分层系统，静默处理冗长 Prompt，仅记录关键检索链路与系统异常。
---

## 🛠️ 技术栈

* **Frontend**: Streamlit
* **Orchestration**: LangChain (Core / Community / Chroma)
* **LLM & Embedding**: Alibaba DashScope (Qwen-Max / Text-Embedding-v4)
* **Vector Database**: ChromaDB (Persistent Storage)
* **Retrieval-Augmented**：BM25 (Keyword Search), Flashrank (Reranking)
---

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名
```
### 2. 安装依赖
```Bash
pip install -r requirements.txt
```
### 3. 配置 API Key
在 config_data.py 中填入你的通义千问
```Bash
API Key：Pythondashscope_api_key = "你的API_KEY"
```
### 4.运行系统录入知识库（上传尺码推荐表等）：
```Bash
streamlit run app_file_uploader.py
```
### 5.启动对话主界面：
```Bash
streamlit run app_qa.py
```
## 📂 目录结构说明
文件名描述
* app_file_uploader.py: 知识库自动化构建界面，处理文件上传与解析 。
* app_qa.py: 智能客服对话前端，支持流式交互 。
* knowledge_base.py: 核心逻辑类，负责文件拆分、MD5 校验与入库 。
* rag.py: RAG 链条逻辑实现，处理检索、Prompt 生成与 LLM 调用 。
* vector_stores.py: 封装向量数据库初始化与检索器配置 。
* config_data.py: 全局配置参数（模型名称、K值、路径等）。

## 📝 许可证
本项目采用 MIT License 开源。
