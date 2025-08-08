# chinese_embedding.py
import numpy as np
import logging
from typing import List, Union, Optional, Dict, Tuple
import os
from pathlib import Path
from config import Config
from logging_utils import setup_logger

paraphrase_multilingual_MiniLM_L12_v2 = "paraphrase-multilingual-MiniLM-L12-v2"
text2vec_base_chinese = "text2vec-base-chinese"
text2vec_large_chinese = "text2vec-large-chinese"
m3e_base = "m3e-base"
m3e_large = "m3e-large"

# 获取默认模型缓存路径
def get_default_model_cache_path():
    """获取默认的模型缓存路径"""
    # 优先使用配置中的路径
    if hasattr(Config, 'MODEL_CACHE_PATH'):
        cache_path = Config.MODEL_CACHE_PATH
    else:
        # 否则使用项目根目录下的models文件夹
        project_root = Path(__file__).parent.parent.parent.parent
        cache_path = os.path.join(project_root, "models")
    
    # 确保目录存在
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

# 统一的模型配置
MODEL_CONFIGS = {
    paraphrase_multilingual_MiniLM_L12_v2: {
        "dimension": 384,
        "description": "多语言轻量级模型，支持中文，性能均衡",
        "library": "sentence-transformers",
        "category": "lightweight"
    },
    text2vec_base_chinese: {
        "dimension": 768, 
        "description": "专门的中文embedding模型，效果优秀",
        "library": "text2vec",
        "category": "balanced"
    },
    text2vec_large_chinese: {
        "dimension": 1024,
        "description": "大型中文embedding模型，效果最佳",
        "library": "text2vec",
        "category": "specialized"
    },
    m3e_base: {
        "dimension": 768,
        "description": "M3E中文embedding模型，针对中文优化",
        "library": "sentence-transformers",
        "model_path": "moka-ai/m3e-base",
        "category": "balanced"
    },
    m3e_large: {
        "dimension": 1024,
        "description": "M3E大型中文embedding模型",
        "library": "sentence-transformers", 
        "model_path": "moka-ai/m3e-large",
        "category": "high_performance"
    }
}

# 按类别组织的模型配置（用于推荐）
CHINESE_EMBEDDING_MODELS = {
    "lightweight": paraphrase_multilingual_MiniLM_L12_v2,
    "balanced": text2vec_base_chinese,
    "high_performance": m3e_large,
    "specialized": text2vec_large_chinese
}

class ChineseEmbeddingFunction:
    """
    中文文本向量化函数类
    支持多种开源中文embedding模型
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu",
                 cache_folder: Optional[str] = None,
                 max_seq_length: int = 512):
        """
        初始化中文embedding函数
        
        Args:
            model_name: 模型名称，支持以下模型:
                - "paraphrase-multilingual-MiniLM-L12-v2": 多语言轻量级模型，384维
                - "text2vec-base-chinese": 专门的中文模型，768维  
                - "text2vec-large-chinese": 大型中文模型，1024维
                - "m3e-base": M3E中文embedding模型，768维
                - "m3e-large": M3E大型中文模型，1024维
            device: 运行设备 ("cpu" 或 "cuda")
            cache_folder: 模型缓存文件夹
            max_seq_length: 最大序列长度
        """
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.cache_folder = cache_folder or get_default_model_cache_path()

        # 设置日志
        self.logger = setup_logger(__name__, log_file='chinese_embedding.log')
        self.logger.info(f"使用模型缓存路径: {self.cache_folder}")
        
        # 验证模型名称
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}")

        self.config = MODEL_CONFIGS[model_name]
        self.dimension = self.config["dimension"]
        
        # 初始化模型
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        加载embedding模型
        根据不同的库类型选择相应的加载方式
        """
        try:
            library = self.config["library"]
            
            if library == "sentence-transformers":
                self._load_sentence_transformer()
            elif library == "text2vec":
                self._load_text2vec_model()
            else:
                raise ValueError(f"不支持的库类型: {library}")
                
            self.logger.info(f"成功加载模型: {self.model_name}")
            self.logger.info(f"模型维度: {self.dimension}")
            self.logger.info(f"设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_sentence_transformer(self):
        """
        加载sentence-transformers模型
        这是最常用的embedding模型库，支持多种预训练模型
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # 确定模型路径
            if "model_path" in self.config:
                model_path = self.config["model_path"]
            else:
                model_path = self.model_name
            
            # 加载模型
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                cache_folder=self.cache_folder
            )
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
                
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
    
    def _load_text2vec_model(self):
        """
        加载text2vec模型
        text2vec是专门针对中文优化的文本向量化库
        """
        try:
            from text2vec import SentenceModel
            
            # text2vec的模型名称映射
            text2vec_models = {
                "text2vec-base-chinese": "shibing624/text2vec-base-chinese",
                "text2vec-large-chinese": "shibing624/text2vec-large-chinese"
            }
            
            model_path = text2vec_models.get(self.model_name, self.model_name)
            
            self.model = SentenceModel(
                model_name_or_path=model_path,
                device=self.device
            )
            
        except ImportError:
            raise ImportError("请安装text2vec: pip install text2vec")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        单个文本向量化
        
        Args:
            text: 输入的中文文本
            
        Returns:
            np.ndarray: 文本的向量表示，形状为(dimension,)
        """
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串类型")
        
        if not text.strip():
            self.logger.warning("输入文本为空，返回零向量")
            return np.zeros(self.dimension, dtype=np.float32)
        
        try:
            # 文本预处理
            processed_text = self._preprocess_text(text)
            
            # 根据模型类型进行编码
            if self.config["library"] == "sentence-transformers":
                embedding = self.model.encode(
                    processed_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # 归一化向量，提高相似度计算准确性
                )
            elif self.config["library"] == "text2vec":
                embedding = self.model.encode(processed_text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            
            # 确保返回正确的数据类型和形状
            embedding = embedding.astype(np.float32)
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"文本向量化失败: {text[:50]}..., 错误: {e}")
            # 返回零向量作为fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    def encode_batch(self, 
                    texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        批量文本向量化
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，根据显存大小调整
            show_progress: 是否显示进度条
            
        Returns:
            np.ndarray: 向量矩阵，形状为(len(texts), dimension)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        
        # 验证输入
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"第{i}个元素不是字符串类型: {type(text)}")
        
        try:
            # 文本预处理
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # 根据模型类型进行批量编码
            if self.config["library"] == "sentence-transformers":
                embeddings = self.model.encode(
                    processed_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress
                )
            elif self.config["library"] == "text2vec":
                # text2vec的批量处理
                embeddings = []
                for i in range(0, len(processed_texts), batch_size):
                    batch = processed_texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(batch)
                    if isinstance(batch_embeddings, list):
                        batch_embeddings = np.array(batch_embeddings)
                    embeddings.append(batch_embeddings)
                
                embeddings = np.vstack(embeddings)
            
            # 确保返回正确的数据类型和形状
            embeddings = embeddings.astype(np.float32)
            
            self.logger.info(f"批量向量化完成: {len(texts)}个文本 -> {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"批量向量化失败: {e}")
            # 返回零向量矩阵作为fallback
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def _preprocess_text(self, text: str) -> str:
        """
        中文文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if not text:
            return ""
        
        # 基础清理
        text = text.strip()
        
        # 移除多余的空白字符
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # 长度截断（根据模型的最大序列长度）
        if len(text) > self.max_seq_length:
            text = text[:self.max_seq_length]
            self.logger.debug(f"文本被截断到{self.max_seq_length}字符")
        
        return text
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 余弦相似度，范围[0, 1]
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            emb1 = self.encode_single(text1).reshape(1, -1)
            emb2 = self.encode_single(text2).reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except ImportError:
            raise ImportError("请安装scikit-learn: pip install scikit-learn")
        except Exception as e:
            self.logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            Dict: 模型详细信息
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
            "description": self.config["description"],
            "library": self.config["library"],
            "cache_folder": self.cache_folder
        }
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        使对象可调用，兼容FAISS工具类的embedding_function参数
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            np.ndarray: 向量或向量矩阵
        """
        if isinstance(text, str):
            return self.encode_single(text)
        elif isinstance(text, list):
            return self.encode_batch(text)
        else:
            raise ValueError(f"不支持的输入类型: {type(text)}")

# 便捷的工厂函数
def create_chinese_embedding_function(model_name: str = paraphrase_multilingual_MiniLM_L12_v2, 
                                    **kwargs) -> ChineseEmbeddingFunction:
    """
    创建中文embedding函数的工厂函数
    
    Args:
        model_name: 模型名称
        **kwargs: 其他参数
        
    Returns:
        ChineseEmbeddingFunction: 配置好的embedding函数
    """
    return ChineseEmbeddingFunction(model_name=model_name, **kwargs)

def get_recommended_model(use_case: str = "balanced") -> Tuple[str, int]:
    """
    根据使用场景推荐模型，并返回模型名称和维度
    
    Args:
        use_case: 使用场景 ("lightweight", "balanced", "high_performance", "specialized")
        
    Returns:
        Tuple[str, int]: 返回(模型名称, 维度)元组
    """
    # 获取推荐的模型名称
    model_name = CHINESE_EMBEDDING_MODELS.get(use_case, CHINESE_EMBEDDING_MODELS["lightweight"])
    
    # 获取模型维度
    dimension = MODEL_CONFIGS[model_name]["dimension"]
    
    return model_name, dimension

def list_available_models() -> Dict[str, Dict]:
    """
    列出所有可用的模型及其详细信息
    
    Returns:
        Dict[str, Dict]: 模型名称到模型详细信息的映射
    """
    result = {}
    for model_name, config in MODEL_CONFIGS.items():
        result[model_name] = {
            "dimension": config["dimension"],
            "description": config["description"],
            "library": config["library"],
            "category": config.get("category", "未分类")
        }
    return result