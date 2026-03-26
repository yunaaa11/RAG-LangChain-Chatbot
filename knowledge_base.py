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
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
import tempfile
from langchain_core.documents import Document
from logger_config import get_logger
from langchain.docstore.base import Docstore
from langchain.storage import create_kv_docstore
logger = get_logger("KnowledgeBase")

def load_document_to_string(file_path):
    """支持多种格式加载并返回纯文本内容"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        # PDF 加载后docs是 List[Document]，需要合并文本
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    elif file_path.endswith('.txt'):
        # 保持原有的文本读取逻辑
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""
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
    #append（追加） 不存在，会自动创建新文件。
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
          # 2. 核心修复点：定义父子分割器 (你之前漏掉了这部分赋值)
        # 子分割器：用于生成存入向量库的索引块
          self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400) 
        # 父分割器：用于生成存入 docstore 的召回块
          self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000) 

        # 3. 方案一实现：使用适配器包装本地存储，解决 memoryview 报错
          parent_cache_path = getattr(config, 'parent_directory', "./parent_folders")
          os.makedirs(parent_cache_path, exist_ok=True)
        
        # 创建原始字节流存储
          _raw_store = LocalFileStore(parent_cache_path)
        
        # 使用 create_kv_docstore 进行自动序列化包装
        # 这样 ParentDocumentRetriever 存入 Document 对象时会自动转为 bytes
          self.store = create_kv_docstore(_raw_store)
    def get_parent_retriever(self):
        """获取父子文档检索器"""
        """此时 self.child_splitter 等属性已正确加载"""
        return ParentDocumentRetriever(
            vectorstore=self.chroma,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
    
    def upload_by_file(self, uploaded_file, filename):
        """处理 PDF 上传，增加指针重置逻辑"""
        # 每一个入口都记录日志
        logger.info(f"收到文件上传请求: {filename}")
        
        if filename.lower().endswith('.pdf'):
        # 关键：确保从文件开头开始读取字节
            uploaded_file.seek(0) 
            # 1. 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read()) # 使用 .read() 获取全部字节
                tmp_path = tmp_file.name
            
            try:
                # 2. 加载 PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                full_text = "\n".join([page.page_content for page in pages])
                logger.info(f"PDF解析成功: {filename}, 共 {len(pages)} 页")
                
                # 3. 调用之前的分段 MD5 逻辑
                return self.upload_by_str(full_text, filename)
            except Exception as e:
                logger.error(f"PDF解析异常: {filename}, 错误: {str(e)}")
                return f"[错误] PDF 解析失败: {str(e)}"
            finally:
                # 4. 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
                # 5. 处理 TXT：直接解码并进行分段校验
                # TXT 处理
            try:
                uploaded_file.seek(0)
                text = uploaded_file.read().decode("utf-8")
                return self.upload_by_str(text, filename)
            except Exception as e:
                logger.error(f"TXT读取异常: {filename}, 错误: {str(e)}")
                return f"[错误] 文件读取失败: {str(e)}"
        
    def upload_by_str(self, data: str, filename):
        """核心逻辑：实现父子文档构建"""
        logger.info(f"开始对 {filename} 进行父子文档索引构建...")
        
        # 1. 计算全文 MD5 校验，防止重复处理整个文件
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            logger.warning(f"[跳过] {filename} 内容已存在")
            return "[跳过] 内容已经存在知识库中"

        try:
            # 2. 构造 Document 对象
            doc = Document(
                page_content=data, 
                metadata={
                    "source": filename,
                    "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "operator": "小曹"
                }
            )
            
            # 3. 使用 ParentDocumentRetriever 添加文档
            # 它会自动完成：父文档存入 self.store，子片段向量化存入 self.chroma
            retriever = self.get_parent_retriever()
            retriever.add_documents([doc], ids=None)
            
            # 4. 记录 MD5
            save_md5(md5_hex)
            
            msg = f"[成功] {filename} 已完成父子文档索引构建"
            logger.info(msg)
            return msg
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            return f"[错误] 索引失败: {str(e)}"
    def get_all_documents(self):
        """获取向量库中所有的原始文档，用于构建 BM25 索引"""
        # 从 Chroma 中提取所有数据并转为 Document 对象
        data = self.chroma.get()
        docs = []
        for content, metadata in zip(data['documents'], data['metadatas']):
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

if __name__=='__main__':
    service=KnowledgeBaseSerivce()
   # 测试代码
    test_text = "这是一段用于测试父子文档检索功能的长文本内容。"
    r = service.upload_by_str(test_text, "test_v1")
    print(r)