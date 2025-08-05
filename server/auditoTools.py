import base64
import numpy as np
import time
import traceback
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import pyaudio
import logging

# FunASR 模型相关
from funasr import AutoModel

# SocketIO 用于前端通信
from flask_socketio import SocketIO

# 日志模块初始化
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ResultDeduplication:
    """
    识别结果去重工具类，防止重复/叠词文本被多次发送
    """
    def __init__(self, max_history=5, similarity_threshold=0.8):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.result_history = deque(maxlen=max_history)
        self.word_cache = set()
        self.last_cache_clear = time.time()

    def is_duplicate(self, new_text):
        """
        判断新识别结果是否为重复内容
        """
        if not new_text or not new_text.strip():
            return True
        new_text = new_text.strip()
        # 与历史结果比对相似度
        for prev_result in self.result_history:
            similarity = self._calculate_similarity(new_text, prev_result['text'])
            if similarity > self.similarity_threshold:
                logger.debug(f"检测到重复内容: {similarity:.2f} - '{new_text}'")
                return True
        # 检查是否为叠词
        words = new_text.split()
        if len(words) > 1:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:
                logger.debug(f"检测到叠词: '{new_text}'")
                return True
        return False

    def add_result(self, text, timestamp=None):
        """
        添加新识别结果到历史记录
        """
        if timestamp is None:
            timestamp = time.time()
        self.result_history.append({
            'text': text.strip(),
            'timestamp': timestamp
        })
        # 定期清空缓存
        if time.time() - self.last_cache_clear > 30:
            self.word_cache.clear()
            self.last_cache_clear = time.time()

    def _calculate_similarity(self, text1, text2):
        """
        计算两个文本的简单词重叠相似度
        """
        if text1 == text2:
            return 1.0
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

class OptimizedWebASR:
    """
    基于静音分段的Web实时语音识别服务
    """
    def __init__(self, buffer_duration=5.0, silence_threshold=0.005, min_voice_len=1.0, tail_silence_len=0.5, socketio=None):
        # 音频参数
        self.RATE = 16000
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.socketio = socketio
        # 分段与静音检测参数
        self.buffer_duration = buffer_duration  # 缓冲区最大长度（秒）
        self.buffer_size = int(self.RATE * buffer_duration)
        self.min_voice_len = int(self.RATE * min_voice_len)      # 最小语音段（秒）
        self.tail_silence_len = int(self.RATE * tail_silence_len)   # 静音检测长度（秒）
        self.silence_threshold = silence_threshold                # 静音能量阈值

        # 音频数据队列和缓冲区
        self.audio_queue = queue.Queue(maxsize=50)
        self.audio_buffer = deque(maxlen=self.buffer_size * 3)
        self.buffer_lock = threading.Lock()

        # 控制与状态变量
        self.last_recognition_time = 0
        self.is_processing = False
        self.is_recording = False
        self.recognition_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

        # 线程池，用于异步识别
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ASR")
        self.recognition_timeout = 10.0

        # 识别结果缓存与去重
        self.recognition_cache = {}
        self.last_cache_clear = time.time()
        self.result_deduplication = ResultDeduplication(
            max_history=5,
            similarity_threshold=0.8
        )
        self.last_processed_audio_hash = None

        # 性能统计
        self.performance_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'duplicate_filtered': 0,
            'average_processing_time': 0,
            'last_recognition_time': 0
        }

        # FunASR模型加载（非流式模型）
        try:
            self.model = AutoModel(
                model="paraformer-zh",   # 非流式模型
                device="cpu",
                batch_size=1,
                disable_log=False
            )
            logger.info("FunASR非流式模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None

        # 启动后台音频处理线程
        self.processing_thread = threading.Thread(
            target=self._background_processor,
            daemon=True,
            name="ASR-Processor"
        )
        self.processing_thread.start()

        # 启动性能监控线程
        self.monitor_thread = threading.Thread(
            target=self._performance_monitor,
            daemon=True,
            name="ASR-Monitor"
        )
        self.monitor_thread.start()
        logger.info(f"WebASR初始化完成 - 静音分段模式")

    def is_silence(self, audio_array):
        """
        判断音频样本是否为静音（RMS能量法）
        """
        audio_float = audio_array.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms < self.silence_threshold

    def add_audio_data(self, audio_data):
        """
        收到前端音频数据，解码并加入队列
        """
        if not self.is_recording:
            return None
        try:
            audio_bytes = base64.b64decode(audio_data)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_array) == 0:
                return None
            logger.debug(f"接收音频数据: {len(audio_array)} 样本")
            # 放入队列，队列满时丢弃最老数据
            try:
                self.audio_queue.put_nowait(audio_array)
            except queue.Full:
                logger.warning("音频队列已满，丢弃旧数据")
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_array)
                except queue.Empty:
                    pass
            return "数据已接收"
        except Exception as e:
            logger.error(f"音频数据处理错误: {e}")
            return None

    def _background_processor(self):
        """
        后台线程：不断获取音频数据，静音检测分段，触发识别
        """
        logger.info("后台处理线程启动 - 静音分段识别模式")
        while True:
            try:
                audio_array = self.audio_queue.get(timeout=1.0)
                with self.buffer_lock:
                    self.audio_buffer.extend(audio_array)
                    current_size = len(self.audio_buffer)
                    # 静音分段判定
                    if current_size >= self.min_voice_len:
                        # 取缓冲区末尾tail_silence_len长度的数据判断是否静音
                        tail = np.array(list(self.audio_buffer)[-self.tail_silence_len:], dtype=np.int16)
                        if self.is_silence(tail):
                            logger.info("检测到静音，触发分段识别")
                            self._async_process_chunk()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"后台处理错误: {e}")
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self._handle_critical_error()
                time.sleep(0.1)

    def _async_process_chunk(self):
        """
        异步方式识别当前缓冲区的音频块
        """
        if self.model is None or self.is_processing:
            return
        self.is_processing = True
        future = self.executor.submit(self._process_streaming_chunk)
        def handle_result(future):
            try:
                result = future.result(timeout=self.recognition_timeout)
                if result and result.strip():
                    if not self.result_deduplication.is_duplicate(result):
                        self.result_deduplication.add_result(result)
                        # 识别结果推送给前端
                        self.socketio.emit('recognition_result', {
                            'text': result,
                            'timestamp': time.time(),
                            'confidence': 0.9,
                            'processing_mode': 'vad'
                        })
                        self.performance_stats['successful_recognitions'] += 1
                        logger.info(f"发送识别结果: {result}")
                    else:
                        self.performance_stats['duplicate_filtered'] += 1
                        logger.debug("最终去重过滤")
                self.consecutive_errors = 0
            except Exception as e:
                logger.error(f"异步识别失败: {e}")
                self.consecutive_errors += 1
            finally:
                self.is_processing = False
        future.add_done_callback(handle_result)

    def _process_streaming_chunk(self):
        """
        实际处理音频块并调用ASR模型
        """
        start_time = time.time()
        self.last_recognition_time = start_time
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.min_voice_len:
                    chunk_data = np.array(list(self.audio_buffer), dtype=np.int16)
                    self.audio_buffer.clear()
                else:
                    return ""
            # 数据归一化并做静音过滤
            audio_float = chunk_data.astype(np.float32) / 32768.0
            audio_rms = np.sqrt(np.mean(audio_float ** 2))
            if audio_rms < self.silence_threshold:
                logger.debug(f"音频信号太弱: {audio_rms:.6f}")
                return ""
            audio_duration = len(audio_float) / self.RATE
            if audio_duration < 1.0:
                logger.debug(f"音频太短: {audio_duration:.2f}秒")
                return ""
            logger.info(f"处理音频块: {audio_duration:.2f}秒, RMS: {audio_rms:.6f}")
            # 调用ASR模型识别
            result = self.model.generate(
                input=audio_float,
                batch_size=1
            )
            if result and len(result) > 0:
                text = result[0].get('text', '').strip() if isinstance(result[0], dict) else str(result[0]).strip()
                processing_time = time.time() - start_time
                self.performance_stats['total_recognitions'] += 1
                self.performance_stats['last_recognition_time'] = processing_time
                logger.info(f"识别成功: '{text}' (耗时: {processing_time:.2f}秒)")
                return text
            return ""
        except Exception as e:
            logger.error(f"识别处理错误: {e}")
            logger.error(traceback.format_exc())
            return ""

    def _performance_monitor(self):
        """
        性能监控线程，定期推送统计信息到前端
        """
        logger.info("性能监控线程启动")
        while True:
            try:
                time.sleep(10)
                stats = self.performance_stats
                if stats['total_recognitions'] > 0:
                    success_rate = stats['successful_recognitions'] / stats['total_recognitions']
                    duplicate_rate = stats['duplicate_filtered'] / stats['total_recognitions']
                    logger.info(f"性能统计 - 总识别: {stats['total_recognitions']}, "
                                f"成功率: {success_rate:.2%}, "
                                f"去重率: {duplicate_rate:.2%}, "
                                f"去重历史: {len(self.result_deduplication.result_history)}")
                    self.socketio.emit('performance_stats', {
                        'total_recognitions': stats['total_recognitions'],
                        'success_rate': success_rate,
                        'duplicate_rate': duplicate_rate,
                        'average_interval': 0,
                        'buffer_size': len(self.audio_buffer)
                    })
            except Exception as e:
                logger.error(f"性能监控错误: {e}")

    def _handle_critical_error(self):
        """
        连续识别错误时，重置状态
        """
        logger.error(f"连续错误达到阈值: {self.consecutive_errors}")
        self.is_processing = False
        self.consecutive_errors = 0
        with self.buffer_lock:
            self.audio_buffer.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("系统状态已重置")

    def start_recording(self):
        """
        开始录音（允许接收音频数据）
        """
        self.is_recording = True
        self.consecutive_errors = 0
        logger.info("开始录音 - 静音分段识别模式")
        return True

    def stop_recording(self):
        """
        停止录音（清空缓冲区与队列）
        """
        self.is_recording = False
        with self.buffer_lock:
            self.audio_buffer.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("停止录音")
        return True

    def get_stats(self):
        """
        获取当前统计信息
        """
        stats = self.performance_stats.copy()
        stats['is_recording'] = self.is_recording
        stats['is_processing'] = self.is_processing
        stats['buffer_size'] = len(self.audio_buffer)
        stats['queue_size'] = self.audio_queue.qsize()
        stats['recognition_interval'] = 0
        return stats