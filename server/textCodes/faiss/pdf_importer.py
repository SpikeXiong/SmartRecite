import os
import re
import logging
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF
from tqdm import tqdm

# 设置日志

def setup_logger(log_file='pdf_importer.log'):
    """设置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

class PDFImporter:
    """PDF导入到FAISS知识库的工具类"""

    def __init__(self, knowledge_base, chunk_size: int = 500, overlap: int = 50, filter_toc: bool = True):
        self.knowledge_base = knowledge_base
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.filter_toc = filter_toc

        if not hasattr(self.knowledge_base, 'embedding_function') or self.knowledge_base.embedding_function is None:
            raise ValueError("知识库必须设置embedding_function才能导入PDF")

    def import_pdf(self, pdf_path: str, password: Optional[str] = None, metadata: Optional[Dict] = None,
                   page_numbers: Optional[List[int]] = None) -> Tuple[int, List[str]]:
        """导入PDF文件到知识库"""
        if not os.path.exists(pdf_path):
            return 0, [f"文件不存在: {pdf_path}"]

        if not pdf_path.lower().endswith('.pdf'):
            return 0, [f"不是PDF文件: {pdf_path}"]

        errors = []
        imported_count = 0

        try:
            doc = self._open_pdf(pdf_path, password)
            if doc is None:
                return 0, ["无法打开PDF文件，可能是密码错误或文件损坏"]

            base_metadata = {
                "source": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "file_type": "pdf",
                "total_pages": len(doc)
            }
            if metadata:
                base_metadata.update(metadata)

            pages_to_process = range(len(doc)) if page_numbers is None else [p for p in page_numbers if 0 <= p < len(doc)]
            if not pages_to_process:
                return 0, ["指定的页码超出PDF范围"]

            toc_pages = self._identify_toc_pages(doc) if self.filter_toc else set()
            logger.info(f"识别到的目录页: {toc_pages}")

            title_hierarchy = self._extract_title_hierarchy(doc)

            for page_num in tqdm(pages_to_process, desc="处理PDF页面"):
                if page_num in toc_pages:
                    logger.info(f"跳过目录页: {page_num + 1}")
                    continue

                page = doc.load_page(page_num)
                text = page.get_text()
                page_title = title_hierarchy.get(page_num, "未定义标题")

                if self.filter_toc:
                    text = self._filter_toc_content(text)

                page_metadata = base_metadata.copy()
                page_metadata["page_number"] = page_num + 1

                chunks = self._chunk_text_by_paragraphs(text)

                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    chunk_metadata = page_metadata.copy()
                    chunk_metadata.update({"title": page_title, "chunk_index": i})

                    try:
                        self.knowledge_base.add_text(chunk, chunk_metadata)
                        imported_count += 1
                    except Exception as e:
                        error_msg = f"添加文本块失败(页码:{page_num + 1}, 块索引:{i}): {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            self.knowledge_base.save()
            logger.info(f"成功导入 {imported_count} 个文本块到知识库")
            return imported_count, errors

        except Exception as e:
            error_msg = f"处理PDF文件时发生错误: {str(e)}"
            logger.error(error_msg)
            return imported_count, [error_msg] + errors

    def _open_pdf(self, pdf_path: str, password: Optional[str] = None) -> Optional[fitz.Document]:
        """打开PDF文件，处理加密情况"""
        try:
            doc = fitz.open(pdf_path)
            if doc.needs_pass and (password is None or not doc.authenticate(password)):
                logger.error(f"PDF文件需要密码或密码错误: {pdf_path}")
                doc.close()
                return None
            return doc
        except Exception as e:
            logger.error(f"打开PDF文件失败: {pdf_path}, 错误: {e}")
            return None

    def _chunk_text_by_paragraphs(self, text: str) -> List[str]:
        """将文本按段分块"""
        if not text:
            return []

        text = self._preprocess_text(text)
        raw_lines = text.split('\n')

        paragraphs = []
        current_para = ""

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            if self._is_title(line):
                if current_para:
                    paragraphs.append(current_para)
                    current_para = ""
                paragraphs.append(line)
                continue

            if len(current_para) + len(line) > self.chunk_size:
                paragraphs.append(current_para)
                current_para = line
            else:
                current_para = f"{current_para} {line}" if current_para else line

        if current_para:
            paragraphs.append(current_para)

        return paragraphs

    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        text = text.replace('\r', '\n')
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return self._filter_watermark(text)

    def _is_title(self, text: str) -> bool:
        """判断文本是否为标题"""
        text = self._filter_watermark(text)
        if len(text) > 50 or not re.search(r'[\u4e00-\u9fff]', text):
            return False

        title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇编]',
            r'^[一二三四五六七八九十\d]+[、.]',
            r'^\(一二三四五六七八九十\d+\)',
        ]
        return any(re.match(pattern, text) for pattern in title_patterns) or len(text) < 20

    def _filter_watermark(self, text: str) -> str:
        """过滤文本中的水印内容"""
        watermark_keywords = [
            r'confidential', r'机密', r'保密文件', r'请勿外传', r'watermark',
            r'版权所有', r'all rights reserved', r'保留所有权利',r'严禁二改二传'
        ]
        lines = text.split('\n')
        return '\n'.join(line for line in lines if not any(re.search(keyword, line.lower()) for keyword in watermark_keywords))

    def _identify_toc_pages(self, doc) -> set:
        """识别PDF中的目录页"""
        toc_pages = set()
        toc = doc.get_toc()
        if toc:
            for page_num in range(min(10, len(doc))):
                text = doc.load_page(page_num).get_text()
                if self._is_toc_page(text):
                    toc_pages.add(page_num)
        return toc_pages

    def _is_toc_page(self, text: str) -> bool:
        """判断文本是否为目录页"""
        toc_keywords = ['目录', '内容', 'contents', 'table of contents', 'toc', 'index']
        text_lower = text.lower()
        page_ref_pattern = r'\.\.\.\s*\d+'
        chapter_pattern = r'^\s*\d+[\.\d]*\s+\w+'
        return any(keyword in text_lower for keyword in toc_keywords) and (
            len(re.findall(page_ref_pattern, text)) > 3 or len(re.findall(chapter_pattern, text, re.MULTILINE)) > 3
        )

    def _filter_toc_content(self, text: str) -> str:
        """过滤目录样式内容"""
        lines = text.split('\n')
        patterns = [
            r'^第\s*[一二三四五六七八九十\d]+\s*[章节篇编]\s*.*\.\.\.\s*\d+$',
            r'^\d+[\.\d]*\s+.*\.\.\.\s*\d+$',
            r'^.*?\.{3,}\s*\d+$'
        ]
        return '\n'.join(line for line in lines if not any(re.match(pattern, line) for pattern in patterns))