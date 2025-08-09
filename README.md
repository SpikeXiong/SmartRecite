
# SmartRecite - 实时语音识别与知识库系统

这是一个基于Flask和SocketIO的实时语音识别系统，使用FunASR模型进行中文语音识别，并支持静音分段和智能去重功能。该系统还集成了基于FAISS的向量检索功能，可用于构建知识库和语义搜索应用，支持PDF文档导入和知识总结。

## 功能特点

- **实时语音识别**：支持浏览器端实时录音并推送到服务器进行识别
- **静音分段识别**：智能检测语音中的静音段，进行自动分段识别
- **结果去重优化**：避免重复内容和叠词被多次识别
- **WebSocket通信**：使用Flask-SocketIO实现前后端实时通信
- **向量检索能力**：集成FAISS向量数据库，支持高效的相似文本检索
- **中文文本向量化**：支持多种中文embedding模型，适应不同应用场景
- **PDF文档导入**：支持导入PDF文档到知识库，自动处理目录和水印
- **知识总结功能**：基于LLM的知识库内容智能总结和问答

## 系统架构

### 核心组件

1. **Flask Web服务器**：提供Web界面和API接口
2. **SocketIO服务**：处理实时双向通信
3. **语音识别模块**：基于FunASR的非流式语音识别
4. **静音检测与分段**：基于能量阈值的语音分段算法
5. **结果去重系统**：基于相似度计算的文本去重
6. **FAISS向量检索**：高性能相似文本检索系统
7. **PDF导入模块**：智能处理PDF文档并分段导入知识库
8. **知识总结模块**：基于大模型的知识库内容智能总结

### 文件结构

```
server/
├── app.py                      # Flask主应用
├── config.py                   # 配置文件
├── logging_utils.py            # 日志工具
├── audio_processing/           # 音频处理相关代码
│   └── auditoTools.py          # 语音识别工具类
├── text_processing/            # 文本处理相关代码
│   ├── faiss/                  # FAISS向量检索相关
│   │   ├── chinese_embedding.py  # 中文文本向量化
│   │   ├── faiss_tools.py        # FAISS知识库工具
│   │   └── __init__.py           # 包初始化
│   ├── file_importer/          # 文件导入相关
│   │   ├── pdf_importer.py     # PDF导入工具
│   │   ├── spacy_splitter.py   # 文本分割工具
│   │   └── __init__.py         # 包初始化
│   ├── knowledge_summarizer.py # 知识总结工具
│   ├── llm_api.py              # LLM API接口
│   └── __init__.py             # 包初始化
├── routes/                     # 路由处理
│   ├── index_routes.py         # 主页路由
│   ├── knowledge_routes.py     # 知识库路由
│   └── __init__.py             # 包初始化
└── templates/                  # 前端模板
    ├── index.html              # 主页面
    ├── knowledge.html          # 知识库页面
    └── test_knowledge_summarizer.html # 知识总结测试页面
```

## 快速开始
推荐Python版本 3.12 / 3.11
### 安装依赖

```bash
# 克隆仓库
git clone hhttps://github.com/SpikeXiong/SmartRecite.git

# 安装依赖
pip install -r requirements.txt
```

### 修改配置

在 `.env` 文件中修改你的配置
```
# API密钥
QWEN_API_KEY=your-api-key-here 
```

### 启动服务器

```bash
python server/app.py
```

服务器将在 http://localhost:8180 启动。

### 使用方法

1. **语音识别功能**:
   - 访问 http://localhost:8180
   - 点击"开始录音"按钮
   - 说话，系统会自动识别并显示结果
   - 点击"停止录音"按钮结束录音

2. **知识库功能**:
   - 访问 http://localhost:8180/knowledge
   - 上传PDF文档到知识库
   - 查看已导入的文档列表
   - 使用搜索功能查询相关内容

3. **知识总结功能**:
   - 访问 http://localhost:8180/test_knowledge_summarizer
   - 输入问题或关键词
   - 系统会自动从知识库中检索相关内容并生成总结

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

### PDF文档处理

PDF导入模块提供以下功能：
- 自动识别和过滤目录页
- 移除页眉页脚和水印
- 智能分段处理文本内容
- 保留文档层次结构信息
- 批量导入目录下的所有PDF文件

### 知识总结功能

知识总结模块基于以下流程：
1. 接收用户查询
2. 从知识库中检索相关内容
3. 使用LLM对检索内容进行分析和总结
4. 生成结构化的回答

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


## 配置参数

主要配置参数位于`.env`中，可以根据需要调整：

```python
# ASR配置
FunASR = "paraformer-zh"  # 语音识别模型
FunASR_DEVICE = "cpu"     # 运行设备

# 知识库配置
KNOWLEDGE_BASE_PATH = "knowledge_base"  # 知识库存储路径
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 向量模型

# LLM配置
QWEN_API_KEY = "your_api_key"  # 千问API密钥
QWEN_API_URL = "https://api.qwen.ai/v1/chat/completions"  # API地址
```

## 贡献

欢迎提交Issue和Pull Request，共同改进这个项目！

## 许可证

MIT License
