import os
from dotenv import load_dotenv
load_dotenv()#加载.env文件内容到环境变量
md5_path="./md5.text"
dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
api_key=dashscope_api_key
embedding_model_name="text-embedding-v4"
chat_model_name="qwen-max"
#Chroma
collection_name="rag"
persist_directory="./chroma_db"
#spliter
chunk_size=1000
chunk_overlap=100
separators=["\n\n","\n",".","!","?","。","!","?"," ","'"]
max_split_char_number=1000 #文本分割器的阈值

operator_name = "小曹"

# 增加返回的片段数量
top_k = 4 
# 设置相似度分数阈值（Chroma通常使用L2距离，数值越小越相似）
score_threshold = 0.5

session_config={
        "configurable":{
            "session_id":"user_001",
        }
    }

# 新增：父文档存储路径（用于 Parent-Document Retrieval）
parent_directory = "./parent_db" 

# 新增：混合检索权重配置
# [向量检索权重, 关键词检索权重]
ensemble_weights = [0.5, 0.5]

# 新增：重排序模型召回数量
rerank_top_k = 5