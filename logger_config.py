import logging
import os
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """配置日志对象，同时输出到文件和控制台"""
    # 确保 logs 文件夹存在
    if not os.path.exists('logs'):
        os.makedirs('logs')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    handler = logging.FileHandler(f'logs/{log_file}', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 防止重复添加 handler
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger

# 创建两个专门的日志记录器
sys_logger = setup_logger('System', 'rag_system.log') # 记录运行状态
prompt_logger = setup_logger('Prompt', 'rag_debug.log') # 专门记录超长的 Prompt