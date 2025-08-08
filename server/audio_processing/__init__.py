"""
音频处理模块
提供实时语音识别、音频缓冲管理和LLM后处理功能
"""

# 导入音频处理工具
from .auditoTools import (
    # 主要类
    OptimizedWebASR,
    AudioBufferManager,
    ResultDeduplication,
    LLMProcessor,
    ASRPerformanceMonitor,
    

)

# 导出所有需要的类和函数
__all__ = [
    # 主要类
    'OptimizedWebASR',
    'AudioBufferManager',
    'ResultDeduplication',
    'LLMProcessor',
    'ASRPerformanceMonitor',
    
]

# 初始化日志
import logging
module_logger = logging.getLogger(__name__)
module_logger.debug("音频处理模块已初始化")