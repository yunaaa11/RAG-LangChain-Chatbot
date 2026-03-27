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
#给这个记录本起个名字叫“VectorStore”
sys_logger = logging.getLogger("VectorStore")

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding 
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
    #Chroma 向量库里存的是“数字（向量）”和“文字（碎片）”。
    #把库里所有的 400 字小块全部“打捞”出来，变成 all_docs 列表，否则下面的 BM25Retriever.from_documents(all_docs) 就没米下锅了。
    def get_all_documents(self):
        data = self.vector_store.get()
        docs = []
        if data and 'documents' in data:
            for content, metadata in zip(data['documents'], data['metadatas']):
                docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def get_retriever(self):
        sys_logger.info("开始构建增强检索器...")
        #向量检索 看语义
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        all_docs = self.get_all_documents()
        sys_logger.info(f"成功获取知识库文档，共计: {len(all_docs)} 条")
        
        if not all_docs:
            return vector_retriever
        #关键词检索 搜“感冒”，它就死磕“感冒”这两个字。如果你文档里写的是“流感”，它可能找不到
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 20
        #混合搜索。确保既能通过意思找，又不会漏掉关键字。 广度，防止漏掉
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], 
            weights=[0.7, 0.3]
        )
        
        try:
        #重排序 (Rerank)
        #TinyBERT 模型把搜回来的 40 条结果，重新拿出来和你的问题进行精读对比。深度，确保给 AI 看精华
        #它把第 15 条（最完美答案）提到第 1 名，并且最后只给你 top_n=3（最准的 3 条）
            sys_logger.info("正在加载 Flashrank Rerank 模型 (TinyBERT)...")
            compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=3)
            #上下文压缩：去粗取精：只把最精华的 3 条推给 AI，扔掉剩下的 37 条垃圾。
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