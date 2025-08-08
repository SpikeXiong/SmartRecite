class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.lastSentTime = 0; // 上次发送数据的时间戳（以秒为单位）
    this.minInterval = 0.01; // 最小发送间隔（秒）
    this.currentTimeFromMainThread = 0; // 从主线程传递的当前时间

    // 监听从主线程传递的消息
    this.port.onmessage = (event) => {
      if (event.data && event.data.currentTime !== undefined) {
        this.currentTimeFromMainThread = event.data.currentTime;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input[0]) {
      const channelData = input[0]; // Float32Array
      const int16Array = new Int16Array(channelData.length);

      for (let i = 0; i < channelData.length; i++) {
        // 将浮点数据转换为 16 位 PCM 数据
        int16Array[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32768));
      }

      // 检查样本数量是否足够
      if (int16Array.length < 16) {
        console.warn('音频数据处理警告: 样本数量过少');
        return true;
      }

      // 确保缓冲区大小正确
      if (int16Array.buffer.byteLength % int16Array.BYTES_PER_ELEMENT === 0) {
        const currentTime = this.currentTimeFromMainThread; // 使用主线程传递的当前时间
        if (currentTime - this.lastSentTime >= this.minInterval) {
          this.port.postMessage({
            buffer: int16Array.buffer,
            length: int16Array.length,
          });
          this.lastSentTime = currentTime;
        }
      } else {
        console.error('音频数据处理错误: 缓冲区大小与元素大小不匹配');
      }
    } else {
      console.error('音频数据处理错误: 输入数据为空');
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);