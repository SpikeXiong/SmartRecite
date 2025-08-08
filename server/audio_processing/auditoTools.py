import base64
import numpy as np
import time
import traceback
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from logging_utils import setup_logger


# FunASR 模型相关
from funasr import AutoModel

from config import Config

#llm
from text_processing import QwenLLMApi

# 日志模块初始化
logger = setup_logger(__name__, log_file='audio_tools.log')

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

class AudioBufferManager:
    """
    音频缓冲与静音检测管理器，负责音频队列、缓冲、分段、静音判断等
    """
    def __init__(self, rate, buffer_duration, min_voice_len, tail_silence_len, silence_threshold):
        self.RATE = rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(self.RATE * buffer_duration)
        self.min_voice_len = int(self.RATE * min_voice_len)
        self.tail_silence_len = int(self.RATE * tail_silence_len)
        self.silence_threshold = silence_threshold

        self.audio_queue = queue.Queue(maxsize=500)
        self.audio_buffer = deque(maxlen=self.buffer_size * 3)
        self.buffer_lock = threading.Lock()

    def add_audio_data(self, audio_array):
        """添加音频数据到队列"""
        try:
            self.audio_queue.put_nowait(audio_array)
        except queue.Full:
            logger.warning("音频队列已满，丢弃旧数据")
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_array)
            except queue.Empty:
                pass

    def get_audio_from_queue(self, timeout=1.0):
        """从队列获取音频数据"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def extend_buffer(self, audio_array):
        """将音频数据加入缓冲区"""
        with self.buffer_lock:
            self.audio_buffer.extend(audio_array)

    def get_current_buffer_size(self):
        """获取当前缓冲区大小"""
        with self.buffer_lock:
            return len(self.audio_buffer)

    def get_tail(self):
        """获取缓冲区尾部用于静音检测"""
        with self.buffer_lock:
            if len(self.audio_buffer) >= self.tail_silence_len:
                return np.array(list(self.audio_buffer)[-self.tail_silence_len:], dtype=np.int16)
            else:
                return np.array([], dtype=np.int16)

    def is_silence(self, audio_array):
        """判断音频样本是否为静音（RMS能量法）"""
        if len(audio_array) == 0:
            return True
        audio_float = audio_array.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms < self.silence_threshold

    def pop_chunk_for_recognition(self):
        """弹出一段用于识别的音频块"""
        with self.buffer_lock:
            if len(self.audio_buffer) >= self.min_voice_len:
                chunk_data = np.array(list(self.audio_buffer), dtype=np.int16)
                self.audio_buffer.clear()
                return chunk_data
            else:
                return None

    def clear(self):
        """清空缓冲区和队列"""
        with self.buffer_lock:
            self.audio_buffer.clear()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

class ASRPerformanceMonitor:
    """
    性能监控线程类，定期推送统计信息到前端
    """
    def __init__(self, performance_stats, result_deduplication, socketio, audio_manager=None, interval=10):
        self.performance_stats = performance_stats
        self.result_deduplication = result_deduplication
        self.socketio = socketio
        self.audio_manager = audio_manager  # 可选，便于获取缓冲区大小
        self.interval = interval
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True, name="ASR-Monitor")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()

    def _monitor(self):
        logger.info("性能监控线程启动")
        while not self._stop_event.is_set():
            try:
                time.sleep(self.interval)
                stats = self.performance_stats
                if stats['total_recognitions'] > 0:
                    success_rate = stats['successful_recognitions'] / stats['total_recognitions']
                    duplicate_rate = stats['duplicate_filtered'] / stats['total_recognitions']
                    buffer_size = self.audio_manager.get_current_buffer_size() if self.audio_manager else 0
                    logger.info(f"性能统计 - 总识别: {stats['total_recognitions']}, "
                                f"成功率: {success_rate:.2%}, "
                                f"去重率: {duplicate_rate:.2%}, "
                                f"去重历史: {len(self.result_deduplication.result_history)}")
                    self.socketio.emit('performance_stats', {
                        'total_recognitions': stats['total_recognitions'],
                        'success_rate': success_rate,
                        'duplicate_rate': duplicate_rate,
                        'average_interval': 0,
                        'buffer_size': buffer_size
                    })
            except Exception as e:
                logger.error(f"性能监控错误: {e}")

class LLMProcessor:
    """
    基于大语言模型的后处理工具类
    """
    def __init__(self, socketio=None):
        self.socketio = socketio
        self.recognized_buffer = ""
        self.buffer_lock = threading.Lock()
        self.QwenLLMApi = QwenLLMApi()
    
    PROCESSED_RESULT = "processed_result"
    MAX_LLM_INPUT_LEN = 2048

    def _llm_proofread(self, text):
        """
        使用大语言模型对文本进行润色或纠错
        """
        logger.info(f"LLM润色处理: {text}")  # 仅打印前50个字符
        prompt = f"以下文本为语音识别的内容，请校对和纠正错误，并且补充必要的标点符号，包括书名号，自动分段，直接输出处理后的结果，不需要任何解释或其他内容：\n{text}\n"
        try:
            # 假设你有 llm_api.chat(prompt) 方法
            response = self.QwenLLMApi.chat(prompt)
            logger.info(f"LLM处理结果: {response}")
            return response
        except Exception as e:
            logger.error(f"LLM润色失败: {e}")
            return text
    
    def _llm_process_and_emit(self, buffer=""):
            """处理并发送LLM处理结果"""
            if not buffer:
                with self.buffer_lock:
                    buffer = self.recognized_buffer
                    self.recognized_buffer = ""
            
            if not buffer.strip():
                return
                
            sentences = self._llm_proofread(buffer)
            self.socketio.emit(LLMProcessor.PROCESSED_RESULT, {
                'text': sentences,
                'timestamp': time.time(),
                'confidence': 0.95,
                'processing_mode': 'llm'
            })
   
    def _llm_timer(self):
        while True:
            time.sleep(3)
            with self.buffer_lock:
                if len(self.recognized_buffer) >= self.MAX_LLM_INPUT_LEN:
                    buffer = self.recognized_buffer
                    self.recognized_buffer = ""
                else:
                    buffer = ""
            if buffer:
                self._llm_process_and_emit(buffer)

    def process_now(self):
        """立即处理当前缓冲区内容（静音等场景可调用）"""
        with self.buffer_lock:
            buffer = self.recognized_buffer
            self.recognized_buffer = ""
        if not buffer.strip():
            return
        for i in range(0, len(buffer), self.MAX_LLM_INPUT_LEN):
            chunk = buffer[i:i+self.MAX_LLM_INPUT_LEN]
            sentences = self._llm_proofread(chunk)
            self.socketio.emit(LLMProcessor.PROCESSED_RESULT, {
                'text': sentences,
                'timestamp': time.time(),
                'confidence': 0.95,
                'processing_mode': 'llm'
                })
            # logger.info(f"LLM处理结果已发送: {sentences}")

    def start(self):
        self.llm_thread = threading.Thread(target=self._llm_timer, daemon=True, name="LLM-Processor")
        self.llm_thread.start()
        logger.info("LLM处理线程启动")

class OptimizedWebASR:
    """
    基于静音分段的Web实时语音识别服务
    """
    def __init__(
        self,
        buffer_duration=5.0,
        silence_threshold=0.005,
        min_voice_len=1.0,
        tail_silence_len=0.5,
        socketio=None
    ):
        self.RATE = 16000
        self.socketio = socketio

        # 统计与去重
        self.performance_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'duplicate_filtered': 0,
            'average_processing_time': 0,
            'last_recognition_time': 0
        }
        self.result_deduplication = ResultDeduplication(
            max_history=5,
            similarity_threshold=0.8
        )

        # 音频缓冲管理
        self.audio_manager = AudioBufferManager(
            rate=self.RATE,
            buffer_duration=buffer_duration,
            min_voice_len=min_voice_len,
            tail_silence_len=tail_silence_len,
            silence_threshold=silence_threshold
        )

        # 性能监控
        self.performance_monitor = ASRPerformanceMonitor(
            performance_stats=self.performance_stats,
            result_deduplication=self.result_deduplication,
            socketio=self.socketio,
            audio_manager=self.audio_manager
        )
        self.performance_monitor.start()

        # LLM处理器
        self.llm_processor = LLMProcessor(socketio=self.socketio)
        self.llm_processor.start()

        # 其它状态
        self.last_recognition_time = 0
        self.is_processing = False
        self.is_recording = False
        self.recognition_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ASR")
        self.recognition_timeout = 5.0
        self.last_processed_audio_hash = None

        # FunASR模型加载
        self._init_model(Config.FunASR, Config.FunASR_DEVICE)

        # 启动后台音频处理线程
        self.processing_thread = threading.Thread(
            target=self._background_processor,
            daemon=True,
            name="ASR-Processor"
        )
        self.processing_thread.start()
        logger.info("WebASR初始化完成 - 静音分段模式")

    def _init_model(self, model_name=Config.FunASR, device=Config.FunASR_DEVICE):
        """动态加载或切换ASR模型"""
        try:
            self.model = AutoModel(
                model=model_name,
                device=device,
                batch_size=1,
                disable_log=False
            )
            logger.info(f"ASR模型切换成功: {model_name} on {device}")
            return True
        except Exception as e:
            logger.error(f"模型切换失败: {e}")
            self.model = None
            return False

    def add_audio_data(self, audio_data):
        """收到前端音频数据，解码并加入队列"""
        if not self.is_recording:
            logger.info("录音未启动，忽略音频数据")
            return None
        try:
            # 检查 Base64 数据长度是否合理
            if len(audio_data) % 4 != 0:
                logger.error("音频数据处理错误: Base64 数据长度不正确")
                return None

            # 解码 Base64 数据
            audio_bytes = base64.b64decode(audio_data)

            # 检查解码后的数据长度是否合理
            if len(audio_bytes) < 32:
                logger.error("音频数据处理错误: 解码后的数据长度过短")
                return None

            # 转换为 numpy 数组
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_array) == 0:
                return None
            
            self.audio_manager.add_audio_data(audio_array)
            return "数据已接收"
        except Exception as e:
            logger.error(f"音频数据处理错误: {e}")
            return None

    def _background_processor(self):
        """后台线程：不断获取音频数据，静音检测分段，触发识别"""
        logger.info("后台处理线程启动 - 静音分段识别模式")
        silence_start_time = None
        asr_silence_threshold = 2.0   # 识别分段静音阈值（秒）
        llm_silence_threshold = 5.0   # LLM校对静音阈值（秒）
        max_buffer_duration = 5.0    # 最大缓冲时长（秒），可根据需求调整
        asr_triggered = False
        llm_triggered = False
        while True:
            try:
                audio_array = self.audio_manager.get_audio_from_queue(timeout=1.0)
                if audio_array is not None:
                    self.audio_manager.extend_buffer(audio_array)
                    current_size = self.audio_manager.get_current_buffer_size()
                    # --- 强制分段逻辑 ---
                    if current_size >= int(self.RATE * max_buffer_duration):
                        logger.info("缓冲区过长，强制触发分段识别，避免数据丢失")
                        self._async_process_chunk()
                        # 这里可以选择是否重置静音计时器
                        silence_start_time = None
                        asr_triggered = False
                        llm_triggered = False
                        continue
                    # --- 原有静音分段逻辑 ---
                    if current_size >= self.audio_manager.min_voice_len:
                        tail = self.audio_manager.get_tail()
                        if self.audio_manager.is_silence(tail):
                            if silence_start_time is None:
                                silence_start_time = time.time()
                                asr_triggered = False
                                llm_triggered = False
                            silence_duration = time.time() - silence_start_time
                            # 语音识别分段
                            if not asr_triggered and silence_duration >= asr_silence_threshold:
                                self._async_process_chunk()
                                asr_triggered = True
                            # LLM校对
                            if not llm_triggered and silence_duration >= llm_silence_threshold:
                                self.llm_processor.process_now()
                                llm_triggered = True
                        else:
                            silence_start_time = None
                            asr_triggered = False
                            llm_triggered = False
            except Exception as e:
                logger.error(f"后台处理错误: {e}")
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self._handle_critical_error()
                time.sleep(0.1)

    def _async_process_chunk(self):
        """异步方式识别当前缓冲区的音频块"""
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
                         # 追加到LLM后处理缓冲区
                        with self.llm_processor.buffer_lock:
                            self.llm_processor.recognized_buffer += result
                        # 推送语音识别结果到前端
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
        """实际处理音频块并调用ASR模型"""
        start_time = time.time()
        self.last_recognition_time = start_time
        try:
            chunk_data = self.audio_manager.pop_chunk_for_recognition()
            if chunk_data is None:
                return ""
            audio_float = chunk_data.astype(np.float32) / 32768.0
            audio_rms = np.sqrt(np.mean(audio_float ** 2))
            if audio_rms < self.audio_manager.silence_threshold:
                logger.debug(f"音频信号太弱: {audio_rms:.6f}")
                return ""
            audio_duration = len(audio_float) / self.RATE
            if audio_duration < 1.0:
                logger.debug(f"音频太短: {audio_duration:.2f}秒")
                return ""
            logger.info(f"处理音频块: {audio_duration:.2f}秒, RMS: {audio_rms:.6f}")
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

    def _handle_critical_error(self):
        """连续识别错误时，重置状态"""
        logger.error(f"连续错误达到阈值: {self.consecutive_errors}")
        self.is_processing = False
        self.consecutive_errors = 0
        self.audio_manager.clear()
        logger.info("系统状态已重置")

    def start_recording(self):
        """开始录音（允许接收音频数据）"""
        self.is_recording = True
        self.consecutive_errors = 0
        logger.info("开始录音 - 静音分段识别模式")
        return True

    def stop_recording(self):
        """停止录音（清空缓冲区与队列）"""
        self.is_recording = False
        self.audio_manager.clear()
        logger.info("停止录音")
        return True

    def get_stats(self):
        """获取当前统计信息"""
        stats = self.performance_stats.copy()
        stats['is_recording'] = self.is_recording
        stats['is_processing'] = self.is_processing
        stats['buffer_size'] = self.audio_manager.get_current_buffer_size()
        stats['queue_size'] = self.audio_manager.audio_queue.qsize()
        stats['recognition_interval'] = 0
        return stats