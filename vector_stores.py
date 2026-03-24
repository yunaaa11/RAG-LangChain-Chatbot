#在线流程向量存储：1.向量化（Embedding）：调用阿里 DashScope 的 text-embedding-v4 模型将文本转为高维向量。
# 2.数据库：使用 ChromaDB 作为本地向量数据库，支持持久化存储，重启后数据不丢失。
# 3.检索器：配置了相似度检索，通过 as_retriever 将用户问题转化为向量并在库中匹配最相关的文本片段。
from langchain_chroma import Chroma
import config_data as config
class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding=embedding 
        self.vector_store=Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
    def get_retriever(self):
        """返回向量检索器,方便加入chain"""
        return self.vector_store.as_retriever(search_kwargs={"k":config.similarity_threshold})
    
if __name__=='__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name,
                                                     dashscope_api_key=config.dashscope_api_key
                                                     )).get_retriever()
    res=retriever.invoke("文档体重180斤,尺码推荐")
    print(res)
    #这里应该是返回所有信息才是对的，文本太少，不能分割，所以所有的尺码当成一个向量存到库里面去了，因此只有1条数据
    