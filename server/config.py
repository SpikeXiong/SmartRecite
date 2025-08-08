# config.py
import os
from pathlib import Path
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent

# 从.env文件加载环境变量
try:
    # 尝试加载.env文件，如果存在的话
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"已从 {env_path} 加载环境变量")
except ImportError:
    print("python-dotenv 未安装，跳过.env文件加载")
    print("可以通过 pip install python-dotenv 安装")

project_root = Path(__file__).parent.parent

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    PORT = int(os.environ.get('PORT', 8180))
    HOST = os.environ.get('HOST', '0.0.0.0')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    KNOWLEDGE_BASE_PATH = os.path.join(project_root, "knowledge_base")
    TEMPLATE_FOLDER = os.path.join(project_root, 'frontend', 'templates')
    STATIC_FOLDER = os.path.join(project_root, 'frontend', 'static')
    
    # 添加更多配置项
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'lightweight')  # 嵌入模型类型
    EMBEDDING_DEVICE = os.environ.get('EMBEDDING_DEVICE', 'cpu')       # 嵌入模型运行设备
    
    # 语音识别配置
    FunASR = os.environ.get('FUN_ASR_MODEL', 'paraformer-zh')
    FunASR_DEVICE = os.environ.get('FUN_ASR_DEVICE', 'cpu')

    # API密钥配置
    QWEN_API_KEY = os.environ.get('QWEN_API_KEY', '')
    QWEN_API_URL = os.environ.get('QWEN_API_URL', 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation')
    QWEN_API_MODEL = os.environ.get('QWEN_API_MODEL', 'qwen-turbo')
    # 语义分段中文模型
    ZH_CORE_WEB_SM = os.environ.get('ZH_CORE_WEB_SM', 'zh_core_web_sm')
    # 模型缓存路径
    MODEL_CACHE_PATH = os.environ.get('MODEL_CACHE_PATH', os.path.join(project_root, "models"))
    