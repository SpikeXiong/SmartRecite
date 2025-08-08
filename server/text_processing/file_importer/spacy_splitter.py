import spacy
from typing import List
from config import Config
ZH_CORE_WEB_SM = Config.ZH_CORE_WEB_SM

class SpacyParagraphSplitter:
    def __init__(self):
        # 加载中文模型
        self.nlp = spacy.load(ZH_CORE_WEB_SM)

    def merge_and_split_paragraphs(self, paragraphs: List[str], max_paragraph_length: int = 300) -> List[str]:
        """先合并段落，然后按语义分割并限制段落长度"""
        # 合并输入段落为一个整体文本
        merged_text = " ".join(paragraphs)
        return self.split_into_paragraphs(merged_text, max_paragraph_length)

    def split_into_paragraphs(self, text: str, max_paragraph_length: int = 300) -> List[str]:
        """使用 spaCy 按语义分割文本，并限制段落长度"""
        paragraphs = []
        doc = self.nlp(text)
        current_para = ""

        for sent in doc.sents:
            sentence = sent.text.strip()
            if not sentence:
                continue

            # 如果当前段落过长，保存并开始新的段落
            if len(current_para) + len(sentence) > max_paragraph_length:
                paragraphs.append(current_para.strip())
                current_para = sentence
            else:
                current_para = f"{current_para} {sentence}" if current_para else sentence

        # 添加最后一个段落
        if current_para:
            paragraphs.append(current_para.strip())

        return paragraphs