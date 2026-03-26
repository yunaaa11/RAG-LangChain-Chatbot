from vector_stores import VectorStoreService, sys_logger
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import config_data as config

class RagService(object):
    def __init__(self, retriever=None):
        # 1. 确定检索器逻辑
        if retriever:
            self.retriever = retriever
            sys_logger.info("RagService 使用传入的增强检索器")
        else:
            vector_service = VectorStoreService(
                embedding=DashScopeEmbeddings(model=config.embedding_model_name,
                                             dashscope_api_key=config.dashscope_api_key)
            )
            self.retriever = vector_service.get_retriever()
            sys_logger.info("RagService 初始化默认检索器")

        # 2. 定义 Prompt 模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system", 
                "你是一个专业客服。请以我提供的已知参考资料为主，简洁和专业的回答用户问题。\n"
                "【参考资料】:\n{context}\n\n"
                "【历史对话记录】:"
            ),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问:{input}")
        ])
        
        self.chat_model = ChatTongyi(model=config.chat_model_name, api_key=config.api_key)
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.retriever

        def format_document(docs: list[Document]):
            if not docs:
                sys_logger.warning("检索阶段未匹配到任何相关文档")
                return "无参考资料" 
            
            # 只保留这一行核心 INFO，记录召回数量
            sys_logger.info(f"大模型输入：已合并 {len(docs)} 条参考片段")
            
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段:{doc.page_content}\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        # 核心 Chain：彻底移除 RunnableLambda(log_prompt)
        chain = ({
            "input": RunnablePassthrough(),
            "context": RunnableLambda(format_for_retriever) | retriever | format_document
        } | RunnableLambda(format_for_prompt_template) 
          | self.prompt_template 
          | self.chat_model 
          | StrOutputParser())

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain