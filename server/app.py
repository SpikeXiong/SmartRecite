from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import logging
import time

# 自定义工具模块
import auditoTools


# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用与SocketIO初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# 全局ASR实例
asr_instance = auditoTools.OptimizedWebASR(
    buffer_duration=5.0,      # 最大缓冲5秒
    silence_threshold=0.005,  # 静音能量阈值
    min_voice_len=1.0,        # 最小语音段1秒
    tail_silence_len=0.5,      # 静音检测0.5秒
    socketio = socketio
)

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    logger.info("启动WebASR服务器 - 静音分段识别模式")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
