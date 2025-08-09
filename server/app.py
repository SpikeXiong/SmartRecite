# 标准库
import logging
import os

# 第三方库
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

# 自定义模块
from audio_processing import OptimizedWebASR
from routes import main_bp, knowledge_bp, register_socketio_handlers
from text_processing import QwenLLMApi
from config import Config
from logging_utils import setup_logger

# 日志配置
logger = setup_logger()

# Flask应用与SocketIO初始化
## 获取项目根目录

app = Flask(
    __name__,
    template_folder=Config.TEMPLATE_FOLDER,
    static_folder=Config.STATIC_FOLDER
)
app.config.from_object(Config)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=1024000)

# 初始化LLM实例
llm = QwenLLMApi()

# 全局ASR实例
asr_instance = OptimizedWebASR(
    buffer_duration=5.0,      # 最大缓冲5秒
    silence_threshold=0.005,  # 静音能量阈值
    min_voice_len=1.0,        # 最小语音段1秒
    tail_silence_len=0.5,     # 静音检测0.5秒
    socketio=socketio
)

# 初始化知识库
def init_knowledge_base():
    """初始化FAISS知识库"""
    try:
        # 导入中文嵌入模型相关函数
        from text_processing.faiss.chinese_embedding import get_recommended_model, create_chinese_embedding_function
        from text_processing.faiss.faiss_tools import PersistentFAISSKnowledgeBase

        # 获取推荐的中文嵌入模型
        embedding_name,  dimension= get_recommended_model(Config.EMBEDDING_MODEL)

        # 创建中文嵌入函数
        embedding_func = create_chinese_embedding_function(
            model_name=embedding_name,
            device=Config.EMBEDDING_DEVICE,
            max_seq_length=512
        )
        
        logger.info(f"中文嵌入模型: {embedding_name}, 嵌入维度: {embedding_func.dimension}")
        
        # 创建知识库实例
        kb = PersistentFAISSKnowledgeBase(
            dimension=dimension,
            storage_path=Config.KNOWLEDGE_BASE_PATH,
            embedding_function=embedding_func
        )
        
        logger.info("知识库初始化成功")
        return kb
    except Exception as e:
        logger.error(f"知识库初始化失败: {str(e)}")
        logger.exception("详细错误信息:")  # 这会记录完整的堆栈跟踪
        return None

# 注册所有蓝图
def register_blueprints(app):
    """注册所有蓝图"""
    app.register_blueprint(main_bp)
    app.register_blueprint(knowledge_bp, url_prefix='/api/knowledge')

# 注册配置
def register_config(app):
    """注册应用配置"""
    # 使用环境变量
    app.config['KNOWLEDGE_BASE'] = init_knowledge_base()
    app.config['LLM_INSTANCE'] = llm


if __name__ == '__main__':
    # 确保知识库目录存在
    os.makedirs(Config.KNOWLEDGE_BASE_PATH, exist_ok=True)

    # 注册配置和蓝图
    register_config(app)
    register_blueprints(app)    

    # 使用routes包中定义的函数注册SocketIO事件处理函数
    register_socketio_handlers(socketio, asr_instance)

    # 减少werkzeug日志输出
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    logger.info("启动WebASR服务器 - 静音分段识别模式")
    logger.info(f"服务器运行在 {Config.HOST}:{Config.PORT}，调试模式: {Config.DEBUG}")
    socketio.run(app, debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
    