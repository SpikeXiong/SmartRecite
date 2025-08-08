from flask import Blueprint, render_template, request, jsonify
from flask_socketio import emit
from logging_utils import setup_logger
import os


# 日志配置
logger = setup_logger(__name__)

# 创建蓝图
main_bp = Blueprint('main', __name__)

# 创建SocketIO事件处理函数集合
socketio_handlers = {}

# 路由
@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/knowledge')
def knowledge_page():
    """知识库管理页面"""
    return render_template('knowledge.html')

@main_bp.route('/test_knowledge_summarizer')
def test_knowledge_summarizer_page():
    """知识点总结测试页面"""
    return render_template('test_knowledge_summarizer.html')

@main_bp.route('/api/knowledge/summarize', methods=['POST'])
def summarize_knowledge():
    """知识点总结接口"""
    try:
        # 获取应用实例
        from flask import current_app
        
        # 获取知识库实例
        knowledge_base = current_app.config.get('KNOWLEDGE_BASE')
        if not knowledge_base:
            return jsonify({'success': False, 'message': '知识库未初始化'})

        # 获取LLM实例
        llm = current_app.config.get('LLM_INSTANCE')
        if not llm:
            return jsonify({'success': False, 'message': 'LLM未初始化'})

        # 获取输入文本
        data = request.json
        input_text = data.get('text', '').strip()
        if not input_text:
            return jsonify({'success': False, 'message': '输入文本不能为空'})

        # 初始化知识点总结器
        from text_processing.knowledge_summarizer import KnowledgeSummarizer
        summarizer = KnowledgeSummarizer(knowledge_base=knowledge_base, llm=llm)

        # 执行知识点总结
        result = summarizer.summarize_knowledge(input_text=input_text)

        return jsonify({'success': True, **result})
    except Exception as e:
        logger.error(f"知识点总结接口错误: {e}")
        return jsonify({'success': False, 'message': str(e)})

# SocketIO事件处理函数
def register_socketio_handlers(socketio, asr_instance):
    """注册所有SocketIO事件处理函数"""
    
    @socketio.on('start_recording')
    def handle_start_recording():
        """
        前端请求：开始录音
        """
        success = asr_instance.start_recording()
        emit('recording_status', {'status': 'started' if success else 'failed'})

    @socketio.on('stop_recording')
    def handle_stop_recording():
        """
        前端请求：停止录音
        """
        success = asr_instance.stop_recording()
        emit('recording_status', {'status': 'stopped' if success else 'failed'})

    @socketio.on('audio_data')
    def handle_audio_data(data):
        """
        前端推送音频数据
        """
        try:
            audio_data = data.get('audio')
            if audio_data:
                result = asr_instance.add_audio_data(audio_data)
                if result is None:
                    emit('error', {'message': '音频处理失败'})
        except Exception as e:
            logger.error(f"处理音频数据错误: {e}")
            emit('error', {'message': str(e)})

    @socketio.on('get_stats')
    def handle_get_stats():
        """
        前端请求：获取统计信息
        """
        stats = asr_instance.get_stats()
        emit('stats_update', stats)

    @socketio.on('connect')
    def handle_connect():
        """
        新客户端连接
        """
        logger.info('客户端已连接')
        emit('status', {
            'message': '连接成功',
            'model_status': 'ready' if asr_instance.model else 'error'
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        """
        客户端断开连接
        """
        asr_instance.is_recording = False
        logger.info(f"客户端断开连接")