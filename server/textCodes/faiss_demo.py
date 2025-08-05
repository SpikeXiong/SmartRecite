import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 初始化嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 示例书本内容
book_content = [
    "机器学习是一种通过数据训练模型的技术。",
    "深度学习是机器学习的一个分支，神经网络是其核心。",
    "监督学习需要标注数据，无监督学习则不需要标注数据。",
]

# 生成嵌入
embeddings = np.array([model.encode(text) for text in book_content])

# 创建 FAISS 索引
dimension = embeddings.shape[1]  # 嵌入向量的维度
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离
index.add(embeddings)  # 录入数据

# 创建一个映射表，用于存储向量 ID 和原始文本
id_to_text = {i: text for i, text in enumerate(book_content)}

# 打印初始数据
print("初始数据：")
for key, value in id_to_text.items():
    print(f"ID: {key}, 内容: {value}")

# -----------------------------------------
# 增加数据
# -----------------------------------------

def add_data(new_text):
    global index, id_to_text
    new_embedding = model.encode(new_text).reshape(1, -1)
    index.add(new_embedding)  # 增加到索引中
    new_id = len(id_to_text)  # 新数据的 ID
    id_to_text[new_id] = new_text  # 更新映射表
    print(f"增加数据：ID: {new_id}, 内容: {new_text}")

# 增加新内容
add_data("强化学习是一种通过奖励和惩罚优化策略的技术。")

# 打印数据
print("\n增加后的数据：")
for key, value in id_to_text.items():
    print(f"ID: {key}, 内容: {value}")

# -----------------------------------------
# 删除数据
# -----------------------------------------

def delete_data(delete_id):
    global index, id_to_text
    if delete_id not in id_to_text:
        print(f"ID: {delete_id} 不存在，无法删除。")
        return
    
    # 删除操作：重新构建索引
    del id_to_text[delete_id]  # 从映射表中删除
    remaining_embeddings = [model.encode(text) for key, text in id_to_text.items()]
    index = faiss.IndexFlatL2(dimension)  # 创建新的索引
    index.add(np.array(remaining_embeddings))  # 重新添加剩余数据
    print(f"删除数据：ID: {delete_id}")

# 删除某条内容
delete_data(1)

# 打印数据
print("\n删除后的数据：")
for key, value in id_to_text.items():
    print(f"ID: {key}, 内容: {value}")

# -----------------------------------------
# 修改数据
# -----------------------------------------

def modify_data(modify_id, new_text):
    global index, id_to_text
    if modify_id not in id_to_text:
        print(f"ID: {modify_id} 不存在，无法修改。")
        return
    
    # 修改操作：更新映射表并重新构建索引
    id_to_text[modify_id] = new_text  # 更新映射表
    updated_embeddings = [model.encode(text) for key, text in id_to_text.items()]
    index = faiss.IndexFlatL2(dimension)  # 创建新的索引
    index.add(np.array(updated_embeddings))  # 重新添加所有数据
    print(f"修改数据：ID: {modify_id}, 新内容: {new_text}")

# 修改某条内容
modify_data(2, "监督学习是一种通过标注数据训练模型的技术。")

# 打印数据
print("\n修改后的数据：")
for key, value in id_to_text.items():
    print(f"ID: {key}, 内容: {value}")

# -----------------------------------------
# 查询数据
# -----------------------------------------

def query_data(query_text, top_k=3):
    query_embedding = model.encode(query_text).reshape(1, -1)
    distances, indices = index.search(query_embedding, k=top_k)  # 检索
    print("\n查询结果：")
    for i, idx in enumerate(indices[0]):
        print(f"排名: {i+1}, ID: {idx}, 内容: {id_to_text[idx]}, 距离: {distances[0][i]}")

# 查询内容
query_data("机器学习的技术有哪些？")
