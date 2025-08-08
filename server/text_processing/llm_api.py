import requests
from config import Config

# 配置
QWEN_API_KEY = Config.QWEN_API_KEY # API Key
QWEN_API_URL = Config.QWEN_API_URL # API URL

class QwenLLMApi:
    """
    通义千问 LLM API 封装，支持本地模型和在线API切换
    """
    def __init__(self, 
                 api_key=QWEN_API_KEY, 
                 api_url=QWEN_API_URL, 
                 use_local=False, 
                 local_model=None):
        """
        :param api_key: 在线API的Key
        :param api_url: 在线API的URL
        :param use_local: 是否使用本地模型
        :param local_model: 本地模型对象（如transformers pipeline）
        """
        self.api_key = api_key
        self.api_url = api_url
        self.use_local = use_local
        self.local_model = local_model

    def chat(self, prompt: str) -> str:
        """
        根据当前模式调用本地或在线千问模型
        """
        if self.use_local:
            return self._chat_local(prompt)
        else:
            return self._chat_online(prompt)

    def _chat_online(self, prompt: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": Config.QWEN_API_MODEL,
            "input": {
                "messages": [
                    {"role": "system", "content": "你是一个用于辅助我记忆知识点的AI助手。"},
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "max_tokens": 2048,
                "temperature": 0.7
            }
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                return result['output']['text'].strip()
            else:
                return f"API调用失败: {resp.status_code}"
        except Exception as e:
            return f"请求出错: {str(e)}"

    def _chat_local(self, prompt: str) -> str:
        if self.local_model is None:
            return "本地模型未初始化"
        try:
            # 假设 local_model 是 transformers pipeline 或类似接口
            result = self.local_model(prompt, max_new_tokens=512)
            # 兼容 transformers pipeline 返回格式
            if isinstance(result, list):
                return result[0]['generated_text']
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
            else:
                return str(result)
        except Exception as e:
            return f"本地模型推理出错: {str(e)}"

# 使用示例
# from llm_api import QwenLLMApi
# llm = QwenLLMApi(
#     api_key='你的API_KEY',
#     api_url='https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
#     use_local=False,
#     local_model=None  # 或传入本地pipeline对象
# )
# response = llm.chat("你好，请自我介绍一下