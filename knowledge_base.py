#知识库自动化构建：1.去重机制：使用 MD5 摘要算法 对文件内容进行哈希计算。
# 如果内容已存在（记录在 md5.text），则跳过处理，避免重复占用向量库空间。
#2.文本切分：利用 RecursiveCharacterTextSplitter 将长文本按段落、句号等符号切分成小块（Chunk），
# 以便模型能更精准地定位信息。
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
def check_md5(md5_str:str):
    '''检查传入的md5字符串是否已经被处理过
    return False(md5未处理过) True(已经处理过，已有记录)
    '''
    if not os.path.exists(config.md5_path):
        #if进入表示文件不存在，肯定没有处理过这个md5
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line=line.strip()#处理字符串前后的空格、回车
            if line==md5_str:
                return True#已经处理过
        return False
    
def save_md5(md5_str:str):
    """将传入md5字符串,记录到文件内保存"""
    with open(config.md5_path,'a',encoding="utf-8") as f:
        f.write(md5_str+'\n')

def get_string_md5(input_str:str,encoding='utf-8'):
    """将传入字符串转换为md5字符串"""
    #将字符串转换为bytes字节数组
    str_bytes=input_str.encode(encoding=encoding)
    #创建md5对象
    md5_obj=hashlib.md5()#得到md5对象
    md5_obj.update(str_bytes)#更新内容（传入即将要转换的字节数组）
    md5_hex=md5_obj.hexdigest()#得到md5的十六进制字符串
    return md5_hex

#知识库更新服务
class KnowledgeBaseSerivce(object):
    def __init__(self):
          #如果文件夹不存在就创建，如果存在就跳过
          os.makedirs(config.persist_directory,exist_ok=True)
          self.chroma=Chroma(
              collection_name=config.collection_name,#数据库的表名
              embedding_function=DashScopeEmbeddings(
                  model="text-embedding-v4",
                  dashscope_api_key=config.dashscope_api_key
                                                     ),
              persist_directory=config.persist_directory,#数据库本地存储文件夹
          ) #向量存储的实例Chroma向量库对象
          self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#分割后的文本段最大长度
            chunk_overlap=config.chunk_overlap,#连线文本段之间的字符重叠数据
            separators=config.separators,#自然段落花费的符号
            length_function=len,#使用python自带len函数
          )#文本分割器的对象
    def upload_by_str(self,data:str,filename):
        """将传入的字符串，进行向量化，存入向量数据库中""" 
        #先得到传入字符串的md5值
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"
        if len(data)>config.max_split_char_number:
            knowledge_chunks:list[str]=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data] # 直接作为一个元素的列表

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小曹"
        }
        self.chroma.add_texts(#内容就加载到向量里
            #iterable->list\tuple
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"

if __name__=='__main__':
    service=KnowledgeBaseSerivce()
    r=service.upload_by_str("小明55","testfile")
    print(r)
    # save_md5("4cf350692a4a3bb54d13daacfe8c683b")
    # print(check_md5("4cf350692a4a3bb54d13daacfe8c683b"))

    # r1=get_string_md5("小明")
    # r2=get_string_md5("小明")
    # r3=get_string_md5("小明11")
    # print(r1)
    # print(r2)
    # print(r3)