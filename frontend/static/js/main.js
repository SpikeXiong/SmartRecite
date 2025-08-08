// WebSocket连接
const socket = io();

// 音频相关变量
let mediaRecorder;
let audioContext;
let analyser;
let microphone;
let dataArray;
let frequencyDataArray;
let processorNode;
let isRecording = false;
let animationId;

// DOM元素
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const results = document.getElementById('results');
const recordingIndicator = document.getElementById('recordingIndicator');
const knowledgePageBtn = document.getElementById('knowledgePageBtn');

// PDF导入相关元素
const pdfImportForm = document.getElementById('pdfImportForm');
const pdfFileInput = document.getElementById('pdfFile');
const fileNameDisplay = document.getElementById('fileName');
const importStatus = document.getElementById('importStatus');
const knowledgeStats = document.getElementById('knowledgeStats');

// 可视化相关元素
const waveformCanvas = document.getElementById('waveformCanvas');
const waveformCtx = waveformCanvas.getContext('2d');
const frequencyCanvas = document.getElementById('frequencyCanvas');
const frequencyCtx = frequencyCanvas.getContext('2d');
const volumeFill = document.getElementById('volumeFill');
const volumeText = document.getElementById('volumeText');
const audioStatus = document.getElementById('audioStatus');
const sampleRateSpan = document.getElementById('sampleRate');
const volumeLevel = document.getElementById('volumeLevel');

// Socket事件监听
socket.on('connect', function() {
    status.textContent = '已连接到服务器';
    status.className = 'status connected';
    
    // 连接后加载知识库统计信息
    fetchKnowledgeStats();
});

socket.on('disconnect', function() {
    status.textContent = '与服务器断开连接';
    status.className = 'status';
});

socket.on('status', function(data) {
    status.textContent = data.message;
    if (data.message.includes('录音')) {
        status.className = 'status recording';
    }
});

// 语音识别结果回调
socket.on('recognition_result', function(data) {
    console.error('收到识别结果:', data);
    addResult(data.text);
});

// LLM校对结果回调
socket.on('processed_result', function(data) {
    console.log('LLM校对结果回调:', data);
    const llmBox = document.getElementById('llmResultBox');
    if (llmBox) {
        llmBox.value += `[${new Date().toLocaleTimeString()}] ${data.text}\n`;
        llmBox.scrollTop = llmBox.scrollHeight;
    }
});

// 初始化可视化
function initVisualization() {
    // 清空波形图
    waveformCtx.fillStyle = '#000';
    waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    
    // 绘制网格
    drawGrid(waveformCtx, waveformCanvas.width, waveformCanvas.height);
    
    // 清空频谱图
    frequencyCtx.fillStyle = '#000';
    frequencyCtx.fillRect(0, 0, frequencyCanvas.width, frequencyCanvas.height);
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    // 垂直网格线
    for (let i = 0; i < width; i += 50) {
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
    }
    
    // 水平网格线
    for (let i = 0; i < height; i += 25) {
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
    }
    
    ctx.stroke();
    
    // 中心线
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
}

// 绘制波形图
function drawWaveform() {
    if (!analyser || !isRecording) return;
    
    analyser.getByteTimeDomainData(dataArray);
    
    // 清空画布
    waveformCtx.fillStyle = '#000';
    waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    
    // 绘制网格
    drawGrid(waveformCtx, waveformCanvas.width, waveformCanvas.height);
    
    // 绘制波形
    waveformCtx.lineWidth = 2;
    waveformCtx.strokeStyle = '#00ff00';
    waveformCtx.beginPath();
    
    const sliceWidth = waveformCanvas.width / dataArray.length;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * waveformCanvas.height / 2;
        
        if (i === 0) {
            waveformCtx.moveTo(x, y);
        } else {
            waveformCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
    }
    
    waveformCtx.stroke();
}

// 绘制频谱图
function drawFrequency() {
    if (!analyser || !isRecording) return;
    
    analyser.getByteFrequencyData(frequencyDataArray);
    
    // 清空画布
    frequencyCtx.fillStyle = '#000';
    frequencyCtx.fillRect(0, 0, frequencyCanvas.width, frequencyCanvas.height);
    
    const barWidth = frequencyCanvas.width / frequencyDataArray.length * 2.5;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < frequencyDataArray.length; i++) {
        barHeight = (frequencyDataArray[i] / 255) * frequencyCanvas.height;
        
        // 根据频率设置颜色
        const hue = (i / frequencyDataArray.length) * 360;
        frequencyCtx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        
        frequencyCtx.fillRect(x, frequencyCanvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
    }
}

// 更新音量显示
function updateVolume() {
    if (!analyser || !isRecording) return;
    
    analyser.getByteTimeDomainData(dataArray);
    
    // 计算RMS音量
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        const sample = (dataArray[i] - 128) / 128;
        sum += sample * sample;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    const volume = Math.round(rms * 100);
    
    // 更新音量条
    volumeFill.style.width = `${Math.min(volume * 2, 100)}%`;
    volumeText.textContent = `音量: ${volume}%`;
    volumeLevel.textContent = `${volume}%`;
}

// 动画循环
function animate() {
    if (!isRecording) return;
    
    drawWaveform();
    drawFrequency();
    updateVolume();
    
    animationId = requestAnimationFrame(animate);
}

// 添加识别结果到页面
function addResult(text) {

    if (results.children.length === 1 && results.children[0].tagName === 'P') {
        results.innerHTML = '';
    }
    
    const resultItem = document.createElement('div');
    resultItem.className = 'result-item';
    
    const timestamp = document.createElement('div');
    timestamp.className = 'timestamp';
    timestamp.textContent = new Date().toLocaleTimeString();
    
    const textDiv = document.createElement('div');
    textDiv.className = 'text';
    textDiv.textContent = text;
    
    resultItem.appendChild(timestamp);
    resultItem.appendChild(textDiv);
    results.appendChild(resultItem);
    
    // 滚动到底部
    results.scrollTop = results.scrollHeight;
}

// 跳转到知识点总结测试页面
const testSummarizerBtn = document.getElementById('testSummarizerBtn');
testSummarizerBtn.addEventListener('click', function() {
    window.location.href = '/test_knowledge_summarizer';
});

// 开始录音
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        audioContext = new AudioContext({ sampleRate: 16000 });
        microphone = audioContext.createMediaStreamSource(stream);

        // 创建分析器节点
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        dataArray = new Uint8Array(analyser.fftSize);
        frequencyDataArray = new Uint8Array(analyser.frequencyBinCount);

        // 加载 AudioWorklet 模块
        await audioContext.audioWorklet.addModule('/static/js/audioProcessor.js');

        // 创建 AudioWorkletNode
        processorNode = new AudioWorkletNode(audioContext, 'audio-processor');
        // 定期发送当前时间到 AudioProcessor
        setInterval(() => {
            processorNode.port.postMessage({ currentTime: audioContext.currentTime });
        }, 10); // 每 10 毫秒发送一次
        processorNode.port.onmessage = (event) => {
            if (event.data && event.data.buffer && event.data.length) {
                try {
                    const uint8Array = new Uint8Array(event.data.buffer);
                    let binaryString = '';
                    for (let i = 0; i < uint8Array.length; i++) {
                        binaryString += String.fromCharCode(uint8Array[i]);
                    }

                    // 转换为 Base64
                    let base64Data = btoa(binaryString);

                    // 检查 Base64 数据长度是否符合要求
                    const paddingLength = 4 - (base64Data.length % 4);
                    if (paddingLength !== 4) {
                        base64Data += '='.repeat(paddingLength);
                        console.warn('Base64 数据填充:', paddingLength, '个 "="');
                    }

                    // 发送数据到服务器
                    socket.emit('audio_data', { audio: base64Data });
                } catch (error) {
                    console.error('音频数据处理错误:', error);
                }
            } else {
                console.error('音频数据处理错误: 接收到的数据不完整');
            }
        };
        // 连接音频节点
        microphone.connect(analyser);
        analyser.connect(processorNode);
        processorNode.connect(audioContext.destination);

        isRecording = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        recordingIndicator.style.display = 'block';

        // 更新状态显示
        audioStatus.textContent = '录音中';
        sampleRateSpan.textContent = `${audioContext.sampleRate}Hz`;

        // 开始动画
        animate();

        socket.emit('start_recording');
    } catch (error) {
        console.error('获取麦克风权限失败:', error);
        alert('无法访问麦克风，请检查权限设置');
    }
}

// 停止录音
function stopRecording() {
    isRecording = false;

    // 停止所有音频节点
    if (microphone) microphone.disconnect();
    if (processorNode) processorNode.disconnect();
    if (audioContext) audioContext.close();

    startBtn.disabled = false;
    stopBtn.disabled = true;
    recordingIndicator.style.display = 'none';

    audioStatus.textContent = '待机';
    socket.emit('stop_recording');
}

// 获取知识库统计信息
function fetchKnowledgeStats() {
    fetch('/api/knowledge/knowledge_stats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const stats = data.stats;
                knowledgeStats.innerHTML = `
                    <strong>知识库统计：</strong> 
                    共有 <b>${stats.total_documents || 0}</b> 个文档，
                    <b>${stats.total_chunks || 0}</b> 个文本块
                `;
            } else {
                knowledgeStats.innerHTML = `<strong>知识库统计：</strong> 获取失败`;
            }
        })
        .catch(error => {
            console.error('获取知识库统计失败:', error);
            knowledgeStats.innerHTML = `<strong>知识库统计：</strong> 获取失败`;
        });
}

// 处理PDF文件选择
pdfFileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        const fileName = this.files[0].name;
        fileNameDisplay.textContent = fileName;
    } else {
        fileNameDisplay.textContent = '未选择文件';
    }
});

// 处理PDF导入表单提交
pdfImportForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('pdfFile');
    if (!fileInput.files || fileInput.files.length === 0) {
        showImportStatus('请选择PDF文件', false);
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const password = document.getElementById('pdfPassword').value;
    if (password) {
        formData.append('password', password);
    }
    
    const title = document.getElementById('pdfTitle').value;
    if (title) {
        formData.append('title', title);
    }
    
    const author = document.getElementById('pdfAuthor').value;
    if (author) {
        formData.append('author', author);
    }
    
    const category = document.getElementById('pdfCategory').value;
    if (category) {
        formData.append('category', category);
    }
    
    // 显示导入中状态
    showImportStatus('正在导入PDF，请稍候...', null);
    
    // 发送请求
    fetch('/api/knowledge/upload_pdf', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showImportStatus(`成功导入 ${data.imported_count} 个文本块`, true);
            
            // 重新获取知识库统计
            fetchKnowledgeStats();
            
            // 重置表单
            pdfImportForm.reset();
            fileNameDisplay.textContent = '未选择文件';
        } else {
            showImportStatus(`导入失败: ${data.message}`, false);
        }
    })
    .catch(error => {
        console.error('PDF导入错误:', error);
        showImportStatus('导入过程中发生错误，请重试', false);
    });
});

// 显示导入状态
function showImportStatus(message, success) {
    importStatus.textContent = message;
    importStatus.style.display = 'block';
    
    if (success === true) {
        importStatus.className = 'import-status success';
    } else if (success === false) {
        importStatus.className = 'import-status error';
    } else {
        importStatus.className = 'import-status';
        importStatus.style.background = '#e9ecef';
        importStatus.style.color = '#495057';
        importStatus.style.border = '1px solid #ced4da';
    }
    
    // 5秒后自动隐藏
    if (success !== null) {
        setTimeout(() => {
            importStatus.style.display = 'none';
        }, 5000);
    }
}

// 跳转到知识库管理页面
knowledgePageBtn.addEventListener('click', function() {
    window.location.href = '/knowledge';
});

// 按钮事件监听
startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    initVisualization();
});

document.getElementById('checkResultsBtn').addEventListener('click', async function () {
    const llmContent = document.getElementById('llmResultBox').value;
    const summarizedResults = document.getElementById('summarizedResults');

    if (!llmContent.trim()) {
        alert('LLM校对结果为空，无法检查结果。');
        return;
    }

    // 清空总结结果区域
    summarizedResults.innerHTML = '<p style="text-align: center; color: #666;">正在检查，请稍候...</p>';

    // 分段处理
    const segments = llmContent.split('\n').filter(line => line.trim());
    const results = [];

    for (const segment of segments) {
        try {
            const response = await fetch('/api/knowledge/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: segment })
            });

            const data = await response.json();
            if (data.success) {
                results.push(`<div class="result-item"><strong>原文:</strong> ${segment}<br><strong>总结:</strong> ${data.summary}</div>`);
            } else {
                results.push(`<div class="result-item"><strong>原文:</strong> ${segment}<br><strong>错误:</strong> 无法生成总结</div>`);
            }
        } catch (error) {
            results.push(`<div class="result-item"><strong>原文:</strong> ${segment}<br><strong>错误:</strong> 请求失败</div>`);
        }
    }

    // 更新总结结果区域
    summarizedResults.innerHTML = results.length
        ? results.join('')
        : '<p style="text-align: center; color: #666;">未生成任何总结结果。</p>';
});