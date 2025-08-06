import logging
from typing import List, Dict, Tuple
from textCodes.faiss.faiss_tools import PersistentFAISSKnowledgeBase
from textCodes.llm_api import QwenLLMApi

# 日志模块初始化
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class KnowledgeSummarizer:
    """
    知识点总结类，结合FAISS知识库和LLM实现。
    """

    def __init__(self, 
                 knowledge_base: PersistentFAISSKnowledgeBase,
                 llm: QwenLLMApi):
        """
        初始化

        Args:
            knowledge_base (PersistentFAISSKnowledgeBase): FAISS知识库实例。
            llm (QwenLLMApi): LLM实例。
        """
        self.knowledge_base = knowledge_base
        self.llm = llm

    def summarize_knowledge(self, input_text: str, max_k: int = 50) -> Dict:
        """
        总结输入文本对应的知识点，并列出未提及的知识点。

        Args:
            input_text (str): 输入文本。
            max_k (int): 检索的最大知识点数量。

        Returns:
            Dict: 包含总结的知识点和未提及的知识点。
        """
        # 检索知识库中的相关知识点
        matched_knowledge = self.knowledge_base.adaptive_search_knowledge(
            query=input_text, max_k=max_k, debug=False
        )
        
        knowledge_result = []
        for text, sim, doc_id, title in matched_knowledge:
             knowledge_result.append({"title": title, "content": text, "similarity": sim})

        logger.info(f"匹配的知识点: {knowledge_result}")
        # 总结匹配到的知识点
        if knowledge_result:
            prompt = f"""
    我需要你作为一个精确的知识匹配专家，执行以下任务：

    1. 我将提供一段用户输入内容和从知识库中检索到的相关知识点列表
    2. 请分析用户输入与知识点的语义关联度，而非仅基于关键词匹配
    3. 从知识库中选择与用户输入最贴切的1-3个知识点
    4. 如果多个知识点相关，请按相关度从高到低排序

    用户输入内容:
    ```
    {input_text}
    ```

    知识库中检索到的相关知识点: 
    ```
    {knowledge_result}
    ```

    【重要规则】:
    - 只返回来自提供知识库的知识点，不要自行创建或修改知识点
    - 对每个选中的知识点，标明其在知识库中的原文
    - 如果知识库中没有真正相关的知识点（相关度低于70%），请直接回复："未找到相关知识点"
    - 不要添加额外解释，只需提供匹配到的知识点原文
    - 如果用户输入涉及多个概念，优先返回覆盖核心概念的知识点
    """
            summary = self.llm.chat(prompt)
        else:
            summary = "未找到相关知识点。"

        # 获取未提及的知识点
        all_knowledge = [self.knowledge_base.get_text_by_id(doc_id) 
                         for doc_id in range(self.knowledge_base.next_id)]
        # unmentioned_knowledge = list(set(all_knowledge) - set(knowledge_result))

        return {
            "summary": summary,
            # "matched_knowledge": knowledge_result,
            # "unmentioned_knowledge": unmentioned_knowledge
        }

# 使用示例
# from faiss.faiss_tools import PersistentFAISSKnowledgeBase
# from llm_api import QwenLLMApi

# 初始化知识库和LLM实例
# knowledge_base = PersistentFAISSKnowledgeBase(embedding_function=your_embedding_function)
# llm = QwenLLMApi(api_key="your_api_key", api_url="your_api_url")

# 初始化知识总结类
# summarizer = KnowledgeSummarizer(knowledge_base=knowledge_base, llm=llm)

# 输入文本
# input_text = "请输入需要总结的文本内容"

# 获取总结结果
# result = summarizer.summarize_knowledge(input_text)
# print(result)