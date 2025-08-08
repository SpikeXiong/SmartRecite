import faiss
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from logging_utils import setup_logger
from config import Config

# 知识库存储路径和索引配置
storage_path = Config.KNOWLEDGE_BASE_PATH
# 索引类型
index_type = "IndexFlatL2"
# 向量维度
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
        self.logger = setup_logger()
    
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
            
            self.logger.info(f"成功添加文本，ID: {doc_id}, 当前索引大小: {self.index.ntotal}, metadata: {metadata}")
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
        self.logger.info(f"开始搜索: {query}, 返回前{k}条结果. 阈值: {threshold}")

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
                if idx >= 0 and idx in self.id_to_text and sim >= threshold:
                    text = self.id_to_text[idx]
                    title = self.metadata[idx].get("title", "未定义标题")
                    results.append((text, float(sim), int(idx), title))
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return []
    
    def adaptive_search_knowledge(self, query: str, max_k: int = 50, debug: bool = True) -> List[Tuple[str, float, int, str]]:
        """
        基于自适应阈值的知识库检索方法。

        该方法会将输入的查询 query 转换为向量，并在 FAISS 索引中检索最相似的 max_k 条记录。
        检索结果会自动过滤无效索引，并根据所有有效结果的相似度均值和标准差，动态计算自适应阈值（均值-0.5*标准差），
        只返回相似度大于等于该阈值的文本内容，最多不超过 max_k 条。

        若 debug=True，会详细打印检索过程和统计信息，便于调试和分析。

        参数:
            query (str): 查询文本，将被转换为向量用于相似度检索。
            max_k (int): 检索时返回的最大候选数量，默认50。
            debug (bool): 是否输出详细调试信息，默认True。

        返回:
            List[Tuple[str, float, int, str]]: 满足自适应阈值的文本内容列表，按相似度降序排列，最多 max_k 条。
        """
        query_embedding = self.embedding_function(query).reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(query_embedding, k=max_k)
        
        if debug:
            print(f"FAISS返回索引: {indices[0]}")
            print(f"FAISS返回相似度: {similarities[0]}")
            print(f"id_to_text字典大小: {len(self.id_to_text)}")
        
        # 分析返回结果
        valid_data = []
        invalid_count = 0
        
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx == -1:
                invalid_count += 1
                if debug:
                    print(f"位置{i}: 无效索引-1 (FAISS填充结果)")
                continue
                
            if idx < 0:
                if debug:
                    print(f"位置{i}: 负索引{idx}")
                continue
                
            if idx not in self.id_to_text:
                if debug:
                    print(f"位置{i}: 索引{idx}不在id_to_text中")
                continue
                
            valid_data.append((idx, sim))
        
        if debug:
            print(f"有效结果数: {len(valid_data)}, 无效结果数: {invalid_count}")
        
        if not valid_data:
            return []
        
        # 基于有效数据计算阈值
        valid_sims = np.array([sim for _, sim in valid_data])
        mean_sim = np.mean(valid_sims)
        std_sim = np.std(valid_sims)
        adaptive_threshold = mean_sim - 0.5 * std_sim
        
        if debug:
            print(f"相似度统计: 均值={mean_sim:.4f}, 标准差={std_sim:.4f}, 阈值={adaptive_threshold:.4f}")
        
        # 构建最终结果
        results = []
        for idx, sim in valid_data:
            if sim >= adaptive_threshold and len(results) < max_k:
                text = self.id_to_text[idx]
                title = self.metadata[idx].get("title", "未定义标题")
                results.append((text, sim, idx, title))
                if debug:
                    print(f"选中结果: 索引={idx}, 相似度={sim:.4f}, 标题={title}")
        
        return results
    
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
    
    def get_all_content(self, page_size: int = 100, page: int = 0, include_text: bool = True) -> Dict:
        """
        分页获取知识库中的所有内容
        
        Args:
            page_size: 每页条目数
            page: 页码（从0开始）
            include_text: 是否包含完整文本内容
            
        Returns:
            Dict: {
                'total': 总条目数,
                'page_count': 总页数,
                'current_page': 当前页码,
                'items': [
                    {
                        'id': 文档ID,
                        'text': 文本内容（如果include_text=True）,
                        'text_preview': 文本预览（始终包含，最多100字符）,
                        'metadata': 元数据
                    },
                    ...
                ]
            }
        """
        try:
            # 获取所有文档ID
            all_ids = list(self.id_to_text.keys())
            total_items = len(all_ids)
            
            if total_items == 0:
                return {
                    'total': 0,
                    'page_count': 0,
                    'current_page': page,
                    'items': []
                }
            
            # 计算总页数
            page_count = (total_items + page_size - 1) // page_size
            
            # 验证页码范围
            if page < 0:
                page = 0
            elif page >= page_count:
                page = page_count - 1
            
            # 计算当前页的数据范围
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, total_items)
            
            # 获取当前页的文档ID
            page_ids = all_ids[start_idx:end_idx]
            
            # 构建结果
            items = []
            for doc_id in page_ids:
                text = self.id_to_text.get(doc_id, "")
                
                # 创建文本预览（最多100字符）
                text_preview = text[:100] + "..." if len(text) > 100 else text
                
                item = {
                    'id': doc_id,
                    'text_preview': text_preview,
                    'metadata': self.metadata.get(doc_id, {})
                }
                
                # 如果需要，添加完整文本
                if include_text:
                    item['text'] = text
                    
                items.append(item)
            
            return {
                'total': total_items,
                'page_count': page_count,
                'current_page': page,
                'items': items
            }
            
        except Exception as e:
            self.logger.error(f"获取知识库内容失败: {e}")
            return {
                'total': 0,
                'page_count': 0,
                'current_page': page,
                'items': [],
                'error': str(e)
            }

# 在PersistentFAISSKnowledgeBase类中添加get_documents方法
def get_documents(self):
    """获取已导入的文档列表"""
    try:
        # 如果你的知识库已经有存储文档元数据的机制，使用它
        # 这里我们假设有一个metadata存储
        if hasattr(self, 'metadata') and self.metadata:
            documents = []
            # 处理元数据，转换为文档列表
            for doc_id, meta in self.metadata.items():
                documents.append({
                    'id': doc_id,
                    'title': meta.get('title', '未命名文档'),
                    'author': meta.get('author', ''),
                    'category': meta.get('category', ''),
                    'chunk_count': meta.get('chunk_count', 0)
                })
            return documents
        else:
            # 如果没有元数据存储，返回空列表
            return []
    except Exception as e:
        print(f"获取文档列表错误: {e}")
        return []

# 在PersistentFAISSKnowledgeBase类中添加get_stats方法的实现
def get_stats(self):
    """获取知识库统计信息"""
    import os
    import datetime
    
    try:
        # 获取索引大小
        index_size = 0
        if os.path.exists(self.storage_path):
            for root, dirs, files in os.walk(self.storage_path):
                for file in files:
                    index_size += os.path.getsize(os.path.join(root, file))
        
        # 获取文档分类
        categories = []
        if hasattr(self, 'metadata') and self.metadata:
            for meta in self.metadata.values():
                if 'category' in meta and meta['category'] not in categories:
                    categories.append(meta['category'])
        
        # 获取最后更新时间
        last_updated = "未知"
        if os.path.exists(self.storage_path):
            last_modified = max(os.path.getmtime(os.path.join(root, file)) 
                               for root, dirs, files in os.walk(self.storage_path) 
                               for file in files) if index_size > 0 else 0
            if last_modified > 0:
                last_updated = datetime.datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建统计信息
        stats = {
            'total_documents': len(self.metadata) if hasattr(self, 'metadata') else 0,
            'total_chunks': self.index.ntotal if hasattr(self, 'index') else 0,
            'index_size': index_size,
            'last_updated': last_updated,
            'categories': categories,
            'documents': self.get_documents() if hasattr(self, 'get_documents') else []
        }
        
        return stats
    except Exception as e:
        print(f"获取知识库统计信息错误: {e}")
        return {
            'total_documents': 0,
            'total_chunks': 0,
            'index_size': 0,
            'last_updated': '未知',
            'categories': [],
            'documents': []
        }