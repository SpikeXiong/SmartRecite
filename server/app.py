from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import logging
import time
import os

# 自定义工具模块
import auditoTools
from textCodes.faiss.faiss_tools import PersistentFAISSKnowledgeBase
from routes.knowledge_routes import knowledge_bp
from routes.index_routes import main_bp
from textCodes.llm_api import QwenLLMApi    

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用与SocketIO初始化
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

llm = QwenLLMApi()

# 全局ASR实例
asr_instance = auditoTools.OptimizedWebASR(
    buffer_duration=5.0,      # 最大缓冲5秒
    silence_threshold=0.005,  # 静音能量阈值
    min_voice_len=1.0,        # 最小语音段1秒
    tail_silence_len=0.5,      # 静音检测0.5秒
    socketio = socketio
)


# 初始化知识库
def init_knowledge_base():
    """初始化FAISS知识库"""
    try:
        # 这里需要实现或导入一个embedding_function
        # 为了简单示例，我们使用随机向量作为临时解决方案
        # 实际应用中应该使用真实的文本向量化模型
        import numpy as np
        def simple_embedding_function(text):
            # 这只是一个示例，实际应用中应该替换为真实的embedding模型
            # 例如使用sentence-transformers等
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(768).astype('float32')
        
        # 创建知识库实例
        kb = PersistentFAISSKnowledgeBase(
            dimension=768,
            storage_path="./knowledge_base",
            embedding_function=simple_embedding_function
        )
        
        logger.info("知识库初始化成功")
        return kb
    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")
        return None

# 注册所有蓝图
def register_blueprints(app):
    """注册所有蓝图"""
    app.register_blueprint(main_bp)
    app.register_blueprint(knowledge_bp, url_prefix='/api/knowledge')

# 注册SocketIO事件处理函数
def register_socketio_handlers(socketio, asr_instance):
    """注册SocketIO事件处理函数"""
    @socketio.on('connect')
    def handle_connect():
        logger.info("客户端已连接")
        emit('response', {'message': '连接成功'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("客户端已断开连接")

    @socketio.on('start_recognition')
    def handle_start_recognition(data):
        """开始识别"""
        logger.info("开始识别请求收到")
        asr_instance.start_recognition()

    @socketio.on('stop_recognition')
    def handle_stop_recognition():
        """停止识别"""
        logger.info("停止识别请求收到")
        asr_instance.stop_recognition()

    @socketio.on('audio_chunk')
    def handle_audio_chunk(data):
        """处理音频数据块"""
        try:
            audio_data = base64.b64decode(data['chunk'])
            asr_instance.process_audio_chunk(audio_data)
        except Exception as e:
            logger.error(f"处理音频数据块时出错: {e}")
            emit('error', {'message': str(e)})

# 注册配置
def register_config(app):
    """注册应用配置"""
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB
    # 将知识库实例存储在应用配置中，以便在路由中访问
    app.config['KNOWLEDGE_BASE'] = init_knowledge_base()
    app.config['LLM_INSTANCE'] = llm



if __name__ == '__main__':
    # 确保知识库目录存在
    os.makedirs("./knowledge_base", exist_ok=True)

    register_config(app)
    register_blueprints(app)
    register_socketio_handlers(socketio, asr_instance)

    logger.info("启动WebASR服务器 - 静音分段识别模式")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)