"""
FAISS知识库模块
提供基于FAISS的向量检索功能和中文文本嵌入能力
"""

# 导入中文嵌入相关功能
from .chinese_embedding import (
    ChineseEmbeddingFunction,
    create_chinese_embedding_function,
    get_recommended_model,

    # 模型常量
    paraphrase_multilingual_MiniLM_L12_v2,
    text2vec_base_chinese,
    text2vec_large_chinese,
    m3e_base,
    m3e_large
)

# 导入FAISS知识库工具
from .faiss_tools import (
    PersistentFAISSKnowledgeBase,
    get_documents,
    get_stats
)

# 为PersistentFAISSKnowledgeBase类添加方法
# 确保这些方法已经在faiss_tools.py中定义
PersistentFAISSKnowledgeBase.get_documents = get_documents
PersistentFAISSKnowledgeBase.get_stats = get_stats

# 导出所有需要的函数和类
__all__ = [
    # 中文嵌入模型
    'ChineseEmbeddingFunction',
    'create_chinese_embedding_function',
    'get_recommended_model',    
    
    # 模型常量
    'paraphrase_multilingual_MiniLM_L12_v2',
    'text2vec_base_chinese',
    'text2vec_large_chinese',
    'm3e_base',
    'm3e_large',
    
    # FAISS知识库
    'PersistentFAISSKnowledgeBase'    
]

# 初始化日志
import logging
logger = logging.getLogger(__name__)
logger.debug("FAISS知识库模块已初始化")