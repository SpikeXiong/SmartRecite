import faiss
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

storage_path = "./knowledge_base"
index_type = "IndexFlatL2"
dimension = 768


class PersistentFAISSKnowledgeBase:
    """持久化FAISS知识库类"""
    
    def __init__(self, 
                 dimension: int = dimension,
                 index_type: str = index_type,
                 storage_path: str = storage_path,
                 embedding_function=None):
        """
        初始化知识库
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 (IndexFlatL2, IndexIVFFlat, IndexHNSWFlat等)
            storage_path: 存储路径
            embedding_function: 文本向量化函数
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = storage_path
        self.embedding_function = embedding_function
        
        # 创建存储目录
        os.makedirs(storage_path, exist_ok=True)
        
        # 初始化索引和数据结构
        self.index = self._create_index()
        self.id_to_text = {}  # ID到文本的映射
        self.text_to_id = {}  # 文本到ID的映射
        self.metadata = {}    # 元数据存储
        self.next_id = 0      # 下一个可用ID
        
        # 文件路径
        self.index_path = os.path.join(storage_path, "faiss.index")
        self.data_path = os.path.join(storage_path, "data.pkl")
        self.config_path = os.path.join(storage_path, "config.json")
        
        # 设置日志
        self._setup_logging()
        
        # 尝试加载已有数据
        self.load()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_index(self):
        """创建FAISS索引"""
        if self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # IVF索引需要训练
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "IndexHNSWFlat":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
    
    def add_text(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        添加文本到知识库
        
        Args:
            text: 要添加的文本
            metadata: 可选的元数据
            
        Returns:
            int: 分配的文档ID
        """
        if not self.embedding_function:
            raise ValueError("未设置embedding_function，无法向量化文本")
        
        # 检查文本是否已存在
        if text in self.text_to_id:
            self.logger.warning(f"文本已存在，ID: {self.text_to_id[text]}")
            return self.text_to_id[text]
        
        try:
            # 生成向量
            embedding = self.embedding_function(text)
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # 确保向量格式正确
            embedding = embedding.reshape(1, -1).astype('float32')
            
            if embedding.shape[1] != self.dimension:
                raise ValueError(f"向量维度不匹配: 期望{self.dimension}, 实际{embedding.shape[1]}")
            
            # 分配ID
            doc_id = self.next_id
            self.next_id += 1
            
            # 添加到索引
            self.index.add(embedding)
            
            # 更新映射关系
            self.id_to_text[doc_id] = text
            self.text_to_id[text] = doc_id
            
            # 存储元数据
            if metadata:
                self.metadata[doc_id] = metadata
            
            self.logger.info(f"成功添加文本，ID: {doc_id}, 当前索引大小: {self.index.ntotal}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"添加文本失败: {e}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        批量添加文本
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            List[int]: 分配的文档ID列表
        """
        if not self.embedding_function:
            raise ValueError("未设置embedding_function，无法向量化文本")
        
        doc_ids = []
        embeddings = []
        new_texts = []
        new_metadatas = []
        
        # 过滤已存在的文本并生成向量
        for i, text in enumerate(texts):
            if text in self.text_to_id:
                doc_ids.append(self.text_to_id[text])
                continue
            
            try:
                embedding = self.embedding_function(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                embedding = embedding.astype('float32')
                if len(embedding.shape) == 1:
                    embedding = embedding.reshape(1, -1)
                
                embeddings.append(embedding)
                new_texts.append(text)
                
                if metadatas and i < len(metadatas):
                    new_metadatas.append(metadatas[i])
                else:
                    new_metadatas.append(None)
                    
                doc_ids.append(self.next_id)
                self.next_id += 1
                
            except Exception as e:
                self.logger.error(f"处理文本失败: {text[:50]}..., 错误: {e}")
                continue
        
        # 批量添加到索引
        if embeddings:
            try:
                batch_embeddings = np.vstack(embeddings)
                self.index.add(batch_embeddings)
                
                # 更新映射关系
                for i, text in enumerate(new_texts):
                    doc_id = doc_ids[len(doc_ids) - len(new_texts) + i]
                    self.id_to_text[doc_id] = text
                    self.text_to_id[text] = doc_id
                    
                    if new_metadatas[i]:
                        self.metadata[doc_id] = new_metadatas[i]
                
                self.logger.info(f"批量添加完成，新增{len(new_texts)}个文档，当前索引大小: {self.index.ntotal}")
                
            except Exception as e:
                self.logger.error(f"批量添加失败: {e}")
                raise
        
        return doc_ids
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float, int]]:
        """
        搜索相似文本
        
        Args:
            query: 查询文本
            k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            List[Tuple[str, float, int]]: (文本, 相似度分数, 文档ID)
        """
        if not self.embedding_function:
            raise ValueError("未设置embedding_function，无法向量化查询")
        
        if self.index.ntotal == 0:
            self.logger.warning("索引为空，无法搜索")
            return []
        
        try:
            # 向量化查询
            query_embedding = self.embedding_function(query)
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # 搜索
            similarities, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
                # 检查索引有效性
                if idx < 0 or idx not in self.id_to_text:
                    continue
                
                # 应用阈值过滤
                if sim >= threshold:
                    text = self.id_to_text[idx]
                    results.append((text, float(sim), int(idx)))
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return []
    
    def get_text_by_id(self, doc_id: int) -> Optional[str]:
        """根据ID获取文本"""
        return self.id_to_text.get(doc_id)
    
    def get_metadata_by_id(self, doc_id: int) -> Optional[Dict]:
        """根据ID获取元数据"""
        return self.metadata.get(doc_id)
    
    def remove_text(self, text: str) -> bool:
        """
        删除文本（注意：FAISS不支持直接删除，这里只是标记删除）
        """
        if text not in self.text_to_id:
            return False
        
        doc_id = self.text_to_id[text]
        
        # 从映射中删除
        del self.text_to_id[text]
        del self.id_to_text[doc_id]
        
        # 删除元数据
        if doc_id in self.metadata:
            del self.metadata[doc_id]
        
        self.logger.info(f"标记删除文档 ID: {doc_id}")
        return True
    
    def save(self) -> bool:
        """
        保存知识库到磁盘
        """
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, self.index_path)
            
            # 保存数据映射和元数据
            data = {
                'id_to_text': self.id_to_text,
                'text_to_id': self.text_to_id,
                'metadata': self.metadata,
                'next_id': self.next_id
            }
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 保存配置
            config = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'created_at': datetime.now().isoformat(),
                'total_documents': len(self.id_to_text)
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"知识库已保存到: {self.storage_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            return False
    
    def load(self) -> bool:
        """
        从磁盘加载知识库
        """
        try:
            # 检查文件是否存在
            if not all(os.path.exists(path) for path in [self.index_path, self.data_path, self.config_path]):
                self.logger.info("未找到已保存的知识库文件，创建新的知识库")
                return False
            
            # 加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 验证配置兼容性
            if config['dimension'] != self.dimension or config['index_type'] != self.index_type:
                self.logger.warning("配置不匹配，将创建新的知识库")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(self.index_path)
            
            # 加载数据
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.id_to_text = data['id_to_text']
            self.text_to_id = data['text_to_id']
            self.metadata = data['metadata']
            self.next_id = data['next_id']
            
            self.logger.info(f"成功加载知识库，文档数量: {len(self.id_to_text)}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载失败: {e}")
            # 重置为空状态
            self.index = self._create_index()
            self.id_to_text = {}
            self.text_to_id = {}
            self.metadata = {}
            self.next_id = 0
            return False
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            'total_documents': len(self.id_to_text),
            'index_size': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'storage_path': self.storage_path,
            'next_id': self.next_id
        }
    
    def clear(self):
        """清空知识库"""
        self.index = self._create_index()
        self.id_to_text = {}
        self.text_to_id = {}
        self.metadata = {}
        self.next_id = 0
        self.logger.info("知识库已清空")

