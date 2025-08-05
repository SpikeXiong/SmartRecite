import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List
import warnings

warnings.filterwarnings("ignore")

# 配置
QWEN_API_KEY = 'sk-322202431aa946b8996d90f0f87f435d'  # 你的阿里云API Key
QWEN_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'

# 初始化嵌入模型
print("正在加载嵌入模型...")
embedding_model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 知识库内容
book_content = [
    "机器学习是一种通过数据训练模型的技术。",
    "深度学习是机器学习的一个分支，神经网络是其核心。",
    "监督学习需要标注数据，无监督学习则不需要标注数据。",
    "强化学习是一种通过奖励和惩罚优化策略的技术。",
    "自然语言处理是人工智能的重要分支，处理人类语言。",
    "计算机视觉专注于让计算机理解和解释视觉信息。",
    "推荐系统根据用户历史行为推荐相关内容。",
    "数据挖掘从大量数据中发现有价值的模式和知识。"
]

# 生成嵌入向量
def generate_embedding(text: str) -> np.ndarray:
    return embedding_model.encode(text, normalize_embeddings=True)

# 构建FAISS索引
print("正在构建索引...")
embeddings = np.array([generate_embedding(text) for text in book_content])
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))
id_to_text = {i: text for i, text in enumerate(book_content)}
print(f"索引构建完成，包含 {index.ntotal} 个向量")

def search_knowledge(query: str, top_k: int = 3) -> List[str]:
    """检索相关知识"""
    query_embedding = generate_embedding(query).reshape(1, -1).astype('float32')
    similarities, indices = index.search(query_embedding, k=top_k)
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        results.append(id_to_text[idx])
    return results

def adaptive_search_knowledge(query: str, max_k: int = 50, debug: bool = True) -> List[str]:
    """自适应检索策略 - 调试版本"""
    query_embedding = generate_embedding(query).reshape(1, -1).astype('float32')
    similarities, indices = index.search(query_embedding, k=max_k)
    
    if debug:
        print(f"FAISS返回索引: {indices[0]}")
        print(f"FAISS返回相似度: {similarities[0]}")
        print(f"id_to_text字典大小: {len(id_to_text)}")
    
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
            
        if idx not in id_to_text:
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
            results.append(id_to_text[idx])
            if debug:
                print(f"选中结果: 索引={idx}, 相似度={sim:.4f}")
    
    return results



def call_qwen_api(prompt: str) -> str:
    """调用阿里云通义千问API"""
    headers = {
        'Authorization': f'Bearer {QWEN_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "max_tokens": 500,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(QWEN_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['output']['text'].strip()
        else:
            return f"API调用失败: {response.status_code}"
    except Exception as e:
        return f"请求出错: {str(e)}"

def answer_question(question: str) -> str:
    """回答问题"""
    # 检索相关知识
    # knowledge = search_knowledge(question, top_k=3)
    knowledge = adaptive_search_knowledge(question)

    # 构建提示
    knowledge_text = "\n".join([f"- {k}" for k in knowledge])
    prompt = f"""用户问题：{question}

相关知识：
{knowledge_text}
重要指令：请严格基于上述知识库内容回答，不要添加知识库中没有明确提及的信息。如果知识库内容不足，请明确说明"根据现有知识库，无法完全回答此问题"。"""
    
    print(f"提示词:{prompt}")
    # 调用API生成回答
    return call_qwen_api(prompt)

def main():
    """主程序"""
    print("\n=== 智能问答系统 ===")
    print("输入 'quit' 退出")
    
    while True:
        question = input("\n请输入问题: ").strip()
        if question.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not question:
            continue
        
        print("正在思考...")
        answer = answer_question(question)
        print(f"\n回答：\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
