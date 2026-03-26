# #在线流程向量存储：1.向量化（Embedding）：调用阿里 DashScope 的 text-embedding-v4 模型将文本转为高维向量。
# # 2.数据库：使用 ChromaDB 作为本地向量数据库，支持持久化存储，重启后数据不丢失。
# # 3.检索器：配置了相似度检索，通过 as_retriever 将用户问题转化为向量并在库中匹配最相关的文本片段。
# from langchain_chroma import Chroma
# import config_data as config
# class VectorStoreService(object):
#     def __init__(self,embedding):
#         self.embedding=embedding 
#         self.vector_store=Chroma(
#             collection_name=config.collection_name,
#             embedding_function=self.embedding,
#             persist_directory=config.persist_directory
#         )
#     def get_retriever(self):
#         """返回向量检索器,方便加入chain"""
#         # 使用 search_type="similarity_score_threshold" 进行分数过滤
#         return self.vector_store.as_retriever(
#             search_type="similarity_score_threshold",# 启用阈值模式
#             search_kwargs={
#                 "k": config.top_k,
#                 "score_threshold": config.score_threshold # 只有相似度够高的才保留
#             }
#         )
    
# if __name__=='__main__':
#     from langchain_community.embeddings import DashScopeEmbeddings
#     retriever=VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name,
#                                                      dashscope_api_key=config.dashscope_api_key
#                                                      )).get_retriever()
#     res=retriever.invoke("文档体重180斤,尺码推荐")
#     print(res)
#     #这里应该是返回所有信息才是对的，文本太少，不能分割，所以所有的尺码当成一个向量存到库里面去了，因此只有1条数据

# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers.document_compressors import FlashrankRerank
# from langchain_chroma import Chroma
# import config_data as config
# import os
# from langchain_core.documents import Document
# import logging

# # 配置日志格式
# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger("VectorStore")

# class VectorStoreService(object):
#     def __init__(self, embedding):
#         self.embedding = embedding 
#         # #在线流程向量存储：1.向量化（Embedding）：调用阿里 DashScope 的 text-embedding-v4 模型将文本转为高维向量。
#         # # 2.数据库：使用 ChromaDB 作为本地向量数据库，支持持久化存储，重启后数据不丢失。
#         self.vector_store = Chroma(
#             collection_name=config.collection_name,
#             embedding_function=self.embedding,
#             persist_directory=config.persist_directory
#         )

#     def get_all_documents(self):
#         """从向量库中提取所有文档，构建 BM25 必须步骤"""
#         data = self.vector_store.get()
#         docs = []
#         if data and 'documents' in data:
#             for content, metadata in zip(data['documents'], data['metadatas']):
#                 docs.append(Document(page_content=content, metadata=metadata))
#         return docs

#     def get_retriever(self):
#         """
#         # 3.检索器：构建混合检索 + Rerank 的增强检索器
#         """
#         # A. 向量检索器 (召回 Top 20)
#         vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        
#         # B. 准备所有文档用于 BM25
#         all_docs = self.get_all_documents()
#         # 埋点日志 1：查看库里有多少数据
#         print(f"--- [LOG] 知识库当前文档总数: {len(all_docs)} ---")
#         # 如果库里没数据，直接返回向量检索器，防止 BM25 报错
#         if not all_docs:
#             logger.warning("知识库为空，仅返回向量检索器")
#             return vector_retriever
            
#         # C. BM25 关键词检索器 (召回 Top 20)
#         bm25_retriever = BM25Retriever.from_documents(all_docs)
#         bm25_retriever.k = 20
        
#         # D. 混合检索 (Ensemble)
#         # 调整权重：0.7 语义 + 0.3 关键词，减少无关“规则”文档的干扰
#         ensemble_retriever = EnsembleRetriever(
#             retrievers=[vector_retriever, bm25_retriever], 
#             weights=[0.7, 0.3]
#         )
        
#         # E. 引入 Rerank (精排)
#         # 使用更轻量的模型并限制最终返回 3 条最相关结果
#         try:
#            # 埋点日志 2：查看 Rerank 是否启动
#             print("--- [LOG] 正在执行 Flashrank 重排序 (Top 3)... ---")
#             compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
#             return ContextualCompressionRetriever(
#                 base_compressor=compressor, 
#                 base_retriever=ensemble_retriever
#             )
#         except Exception as e:
#             print(f"--- [ERROR] Rerank 失败: {e} ---")
#             return ensemble_retriever
    
# if __name__ == '__main__':
#     from langchain_community.embeddings import DashScopeEmbeddings
#     service = VectorStoreService(DashScopeEmbeddings(
#         model=config.embedding_model_name,
#         dashscope_api_key=config.dashscope_api_key
#     ))
#     enhanced_retriever = service.get_retriever()
#     res = enhanced_retriever.invoke("文档体重180斤,尺码推荐")
#     for i, doc in enumerate(res):
#         print(f"排名 {i+1} 内容: {doc.page_content[:50]}...")
# vector_stores.py
import logging
import os
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
import config_data as config
from langchain_core.documents import Document

# 统一配置：只往 rag_system.log 写 INFO 级别日志
if not os.path.exists('logs'):
    os.makedirs('logs')

# 只要这一处配置就够了
logging.basicConfig(
    filename='logs/rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
sys_logger = logging.getLogger("VectorStore")

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding 
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )

    def get_all_documents(self):
        data = self.vector_store.get()
        docs = []
        if data and 'documents' in data:
            for content, metadata in zip(data['documents'], data['metadatas']):
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def get_retriever(self):
        # 以下就是你想要的那几行 INFO
        sys_logger.info("开始构建增强检索器...")
        
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        all_docs = self.get_all_documents()
        sys_logger.info(f"成功获取知识库文档，共计: {len(all_docs)} 条")
        
        if not all_docs:
            return vector_retriever
            
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 20
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], 
            weights=[0.7, 0.3]
        )
        
        try:
            sys_logger.info("正在加载 Flashrank Rerank 模型 (TinyBERT)...")
            compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=ensemble_retriever
            )
            sys_logger.info("增强检索器 (Hybrid + Rerank) 初始化完成")
            return compression_retriever
        except Exception as e:
            sys_logger.error(f"Rerank 失败: {e}")
            return ensemble_retriever
    
if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    # 初始化测试
    service = VectorStoreService(DashScopeEmbeddings(
        model=config.embedding_model_name,
        dashscope_api_key=config.dashscope_api_key
    ))
    enhanced_retriever = service.get_retriever()
    res = enhanced_retriever.invoke("文档体重180斤,尺码推荐")
    for i, doc in enumerate(res):
        print(f"排名 {i+1} 内容: {doc.page_content[:50]}...")