"""
路由包初始化文件
这个包包含了应用的所有路由蓝图和SocketIO事件处理函数
"""

# 导出主要蓝图，使它们可以从routes包直接导入
from .index_routes import main_bp, register_socketio_handlers
from .knowledge_routes import knowledge_bp

# 导出所有需要的函数和变量
__all__ = [
    'main_bp',              # 主蓝图，包含基本页面路由
    'knowledge_bp',         # 知识库蓝图，包含知识库相关API
    'register_socketio_handlers'  # SocketIO事件处理函数注册器
]

# 包初始化日志
import logging
logger = logging.getLogger(__name__)
logger.debug("路由模块已初始化")