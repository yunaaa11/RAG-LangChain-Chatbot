#会话状态与历史管理:1.本地化存储：不同于内存存储，该项目将聊天记录以 JSON 格式持久化在本地磁盘。
#2.会话隔离：通过 session_id 区分不同用户的聊天历史。
import os,json
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage,message_to_dict,messages_from_dict
def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id=session_id#会话id
        self.storage_path=storage_path #不同会话Id存储文件、所在的文件夹路径
        #完整的文件路径
        self.file_path=os.path.join(self.storage_path,self.session_id)
        #不同用户聊天记录存在不同地方 互不干扰

        #确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
    def add_messages(self,messages:Sequence[BaseMessage])->None:
        #Sequence序列 类似list、tuple 写信给ai
        all_messages=list(self.messages)#已有的消息列表
        all_messages.extend(messages)#新的和已有的融合成list
        new_messages=[message_to_dict(message) for message in all_messages]
        #将数据写入文件
        with open(self.file_path,'w',encoding="utf-8") as f:
            json.dump(new_messages,f)
        
    @property#转授权将messages方法变成成员属性用
    def messages(self)->list[BaseMessage]:
        #当前文件内:list[字典] ai读信
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data=json.load(f)#返回值是:list[字典]
                return messages_from_dict(messages_data)
            #将 JSON 文件中的消息数据，转换为 LangChain 可识别的消息对象列表（HumanMessage 或 AIMessage 对象）。
        except FileNotFoundError:
            return []
        
    def clear(self)->None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)

