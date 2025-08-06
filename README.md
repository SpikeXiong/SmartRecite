# Audito - 实时语音识别系统

Audito是一个基于Flask和SocketIO的实时语音识别系统，使用FunASR模型进行中文语音识别，并支持静音分段和智能去重功能。该系统还集成了基于FAISS的向量检索功能，可用于构建知识库和语义搜索应用。

## 功能特点

- **实时语音识别**：支持浏览器端实时录音并推送到服务器进行识别
- **静音分段识别**：智能检测语音中的静音段，进行自动分段识别
- **结果去重优化**：避免重复内容和叠词被多次识别
- **WebSocket通信**：使用Flask-SocketIO实现前后端实时通信
- **向量检索能力**：集成FAISS向量数据库，支持高效的相似文本检索
- **中文文本向量化**：支持多种中文embedding模型，适应不同应用场景

## 系统架构

### 核心组件

1. **Flask Web服务器**：提供Web界面和API接口
2. **SocketIO服务**：处理实时双向通信
3. **语音识别模块**：基于FunASR的非流式语音识别
4. **静音检测与分段**：基于能量阈值的语音分段算法
5. **结果去重系统**：基于相似度计算的文本去重
6. **FAISS向量检索**：高性能相似文本检索系统

### 文件结构

```
server/
├── app.py                 # Flask主应用
├── auditoTools.py         # 语音识别工具类
├── textCodes/             # 文本处理相关代码
│   ├── faiss/             # FAISS向量检索相关
│   │   ├── chinese_embedding.py  # 中文文本向量化
│   │   ├── faiss_tools.py        # FAISS知识库工具
│   │   └── test_faiss.py         # FAISS测试代码
│   ├── faiss_demo.py      # FAISS基础示例
│   └── faiss_llm_demo.py  # FAISS与LLM集成示例
└── templates/             # 前端模板
    └── index.html         # 主页面
```

## 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/audito.git
cd audito

# 安装依赖
pip install -r requirements.txt
```

### 启动服务器

```bash
cd server
python app.py
```

服务器将在 http://localhost:5000 启动。

### 使用方法

1. 访问 http://localhost:5000
2. 点击"开始录音"按钮
3. 说话，系统会自动识别并显示结果
4. 点击"停止录音"按钮结束录音

## 技术实现

### 语音识别流程

1. 浏览器通过WebSocket将音频数据发送到服务器
2. 服务器将音频数据添加到缓冲区
3. 系统检测静音段，触发识别
4. FunASR模型进行语音识别
5. 结果去重系统过滤重复内容
6. 通过WebSocket将识别结果实时返回给前端

### 静音分段算法

系统使用基于能量阈值的静音检测算法：
- `silence_threshold`: 静音能量阈值
- `min_voice_len`: 最小语音段长度
- `tail_silence_len`: 尾部静音检测长度

当检测到足够长的静音段时，系统会自动触发识别过程。

### 结果去重机制

系统使用以下策略避免重复识别结果：
- 历史结果缓存与相似度比较
- 叠词检测
- 词重叠率计算

## 向量检索功能

系统集成了FAISS向量检索功能，支持：

1. **中文文本向量化**：
   - 支持多种中文embedding模型
   - 提供轻量级、平衡型和高性能模型选择

2. **持久化知识库**：
   - 支持知识库的保存和加载
   - 提供文档增删改查功能

3. **相似文本检索**：
   - 基于余弦相似度的文本检索
   - 支持自适应阈值过滤

## 示例：构建知识库

```python
from server.textCodes.faiss.chinese_embedding import create_chinese_embedding_function
from server.textCodes.faiss.faiss_tools import PersistentFAISSKnowledgeBase

# 创建embedding函数
embedding_func = create_chinese_embedding_function()

# 创建知识库
kb = PersistentFAISSKnowledgeBase(
    dimension=embedding_func.dimension,
    embedding_function=embedding_func
)

# 添加文档
kb.add_texts(["文档1", "文档2", "文档3"])

# 搜索相似文档
results = kb.search("查询文本", k=3)
```

## 性能优化

系统包含多项性能优化措施：
- 使用线程池进行异步识别
- 批量处理音频数据
- 缓冲区大小自适应调整
- 错误重试与恢复机制

## 配置参数

主要配置参数位于`app.py`中：

```python
asr_instance = auditoTools.OptimizedWebASR(
    buffer_duration=5.0,      # 最大缓冲5秒
    silence_threshold=0.005,  # 静音能量阈值
    min_voice_len=1.0,        # 最小语音段1秒
    tail_silence_len=0.5,     # 静音检测0.5秒
    socketio=socketio
)
```

## 贡献

欢迎提交Issue和Pull Request，共同改进这个项目！

## 许可证

MIT License