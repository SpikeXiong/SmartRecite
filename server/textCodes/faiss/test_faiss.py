import chinese_embedding
import faiss_tools

if __name__ == "__main__":
    embedding_name = chinese_embedding.get_recommended_model("lightweight")
    
    embedding_func = chinese_embedding.create_chinese_embedding_function(
        model_name = embedding_name,
        device="cpu",
        max_seq_length=512)
    
    storage_path: str = "./test_chinese_kb"

    #创建知识库
    kb = faiss_tools.PersistentFAISSKnowledgeBase(embedding_func.dimension, "IndexFlatL2", storage_path, embedding_func)

    print(f"创建中文知识库成功:")
    print(f"模型: {embedding_name}")
    print(f"维度: {embedding_func.dimension}")
    print(f"存储路径: {storage_path}")

    # 添加中文文档
    chinese_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
        "机器学习是人工智能的核心，是使计算机具有智能的根本途径。",
        "深度学习是机器学习的一个分支，它基于人工神经网络进行学习。",
        "自然语言处理是人工智能的重要应用领域，让计算机理解人类语言。",
        "计算机视觉让机器能够识别和理解图像和视频内容。",
        "知识图谱是一种结构化的知识表示方式，描述实体及其关系。",
        "强化学习通过与环境交互来学习最优的行为策略。",
        "大语言模型如GPT和BERT在自然语言理解方面取得了突破性进展。"
    ]

    doc_ids = kb.add_texts(chinese_texts)
    print(f"添加了 {len(doc_ids)} 个中文文档")

    # 中文查询测试
    queries = [
        "什么是人工智能？",
        "机器学习的原理",
        "深度学习技术",
        "NLP自然语言处理"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = kb.search(query, k=3, threshold=0.3)
        
        for i, (text, score, doc_id) in enumerate(results, 1):
            print(f"  {i}. 相似度: {score:.3f}")
            print(f"     文档: {text}")
    
    # 保存知识库
    kb.save()
    print(f"\n知识库已保存")
    
    # 显示统计信息
    stats = kb.get_stats()
    print(f"\n知识库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
