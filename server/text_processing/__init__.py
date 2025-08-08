"""
文本处理模块
提供文本向量化、知识库管理和文件导入等功能
"""

# 导入子包
from . import faiss
from . import file_importer

# 从子包中导入特定组件
from .faiss import (
    ChineseEmbeddingFunction, create_chinese_embedding_function,
    get_recommended_model, 
    paraphrase_multilingual_MiniLM_L12_v2, text2vec_base_chinese,
    text2vec_large_chinese, m3e_base, m3e_large,
    PersistentFAISSKnowledgeBase
)

from .file_importer import (
    PDFImporter, SpacyParagraphSplitter
)

from .knowledge_summarizer import KnowledgeSummarizer
from .llm_api import QwenLLMApi

# 定义__all__列表
__all__ = [
    # 子包
    'faiss', 'file_importer',

    # 嵌入相关
    'ChineseEmbeddingFunction', 'create_chinese_embedding_function',
    'get_recommended_model',
    'paraphrase_multilingual_MiniLM_L12_v2', 'text2vec_base_chinese',
    'text2vec_large_chinese', 'm3e_base', 'm3e_large',

    # 持久化知识库
    'PersistentFAISSKnowledgeBase',
    # 文件导入
    'PDFImporter', 'SpacyParagraphSplitter',
    
    # 知识点总结
    'KnowledgeSummarizer', 
    # Qwen LLM API
    'QwenLLMApi'
]

# 初始化日志
import logging
logger = logging.getLogger(__name__)
logger.debug("文本处理模块已初始化")