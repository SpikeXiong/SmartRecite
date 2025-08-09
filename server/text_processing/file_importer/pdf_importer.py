import os
import re
import logging
from typing import List, Dict, Optional, Tuple
import fitz 
from tqdm import tqdm
from logging_utils import setup_logger

from .spacy_splitter import SpacyParagraphSplitter


# 设置日志
logger = setup_logger(__name__, log_file='pdf_importer.log')

class PDFImporter:
    """PDF导入到FAISS知识库的工具类"""

    def __init__(self, knowledge_base, chunk_size: int = 500, overlap: int = 50, filter_toc: bool = True):
        self.knowledge_base = knowledge_base
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.filter_toc = filter_toc
        self.splitter = SpacyParagraphSplitter() # 使用spaCy进行段落分割

        # 检查知识库是否设置了embedding_function
        if not hasattr(self.knowledge_base, 'embedding_function') or self.knowledge_base.embedding_function is None:
            raise ValueError("知识库必须设置embedding_function才能导入PDF")

    def import_pdf(self, pdf_path: str, password: Optional[str] = None, metadata: Optional[Dict] = None,
                   page_numbers: Optional[List[int]] = None) -> Tuple[int, List[str]]:
        """导入PDF文件到知识库，并根据文本块提取目录"""
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

            # 提取所有页面的文本并合并
            all_text = ""
            for page_num in tqdm(pages_to_process, desc="提取PDF文本"):
                if page_num in toc_pages:
                    continue
                
                page = doc.load_page(page_num)
                # 使用高级文本提取以获取更多细节信息
                page_dict = page.get_text("dict")
                filtered_text = ""
                
                # 过滤水印内容
                for block in page_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = ""
                            
                            for span in line["spans"]:
                                # 使用非正交旋转检测过滤水印
                                if "bbox" in span and self._is_non_orthogonal_rotation(span["bbox"]):
                                    logger.info(f"检测到可能的水印文本: {span.get('text', '')}")
                                    break

                                line_text += span.get("text", "")
                            
                            if line_text.strip():
                                filtered_text += line_text + "\n"
                
                logger.info(f"过滤后水印的文本:{filtered_text}")
                page_text = filtered_text

                # 过滤目录内容
                if self.filter_toc:                    
                    page_text = self._filter_toc_content(page_text)
                    logger.info(f"过滤后目录的文本:{page_text}")

                
                # 添加页码标记以便后续处理
                all_text += f"\n[PAGE_BREAK_{page_num+1}]\n{page_text}"

            # 预处理合并后的文本
            processed_text = self._preprocess_text(all_text)

            # 分块处理并提取目录
            chunks, title_hierarchy = self._chunk_text_with_context(processed_text)

            # 添加到知识库
            for i, chunk_info in enumerate(chunks):
                chunk_text, page_num = chunk_info['text'], chunk_info['page']
                title_key = chunk_info.get('title_key', 'default')
                # 获取标题信息，如果找不到则使用默认值
                page_title = title_hierarchy.get(title_key, {"title": "未定义标题", "full_path": "", "level": 0})
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "title": page_title['title'],
                    "title_path": page_title['full_path'],
                    "title_level": page_title['level']
                })
                try:
                    self.knowledge_base.add_text(chunk_text, chunk_metadata)
                    imported_count += 1
                except Exception as e:
                    error_msg = f"添加文本块失败(页码:{page_num + 1}, 块索引:{i}): {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            self.knowledge_base.save()
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

    def _chunk_text_with_context(self, text: str) -> Tuple[List[Dict], Dict]:
        """按页码标记分割文本，并根据标题动态生成目录层次"""
        # 使用更精确的正则表达式匹配页码标记
        page_breaks = re.findall(r'\n\[PAGE_BREAK_(\d+)\]\n', text)
        
        if not page_breaks:
            logger.warning("未找到任何页码标记，无法分割文本")
            return [], {}
        
        # 初始化数据结构
        chunks = []
        title_hierarchy = {}
        current_titles = [[] for _ in range(10)]  # 为每个级别创建一个标题列表
        title_counter = {}  # 用于跟踪每页的标题计数
        
        # 构建页面内容字典
        page_contents = self._extract_page_contents(text, page_breaks)
        
        # 处理每个页面
        for page_num_str in page_breaks:
            try:
                page_num = int(page_num_str)
                
                # 初始化当前页的标题计数
                if page_num not in title_counter:
                    title_counter[page_num] = 0
                    
                # 获取页面内容
                page_content = page_contents.get(page_num)
                if not page_content:
                    continue
                    
                # 处理页面内容
                page_chunks, page_titles = self._process_page_content(
                    page_content, page_num, title_counter, current_titles, title_hierarchy
                )
                
                # 更新结果
                chunks.extend(page_chunks)
                title_hierarchy.update(page_titles)
                
            except Exception as e:
                logger.error(f"处理页码 {page_num_str} 时出错: {str(e)}")
                continue
        
        return chunks, title_hierarchy

    def _extract_page_contents(self, text: str, page_breaks: List[str]) -> Dict[int, str]:
        """从文本中提取每个页面的内容"""
        page_contents = {}
        
        for i, page_num_str in enumerate(page_breaks):
            page_num = int(page_num_str)
            marker = f"\n[PAGE_BREAK_{page_num}]\n"
            start_pos = text.find(marker)
            
            if start_pos == -1:
                logger.error(f"无法找到页码标记: {marker}")
                continue
                
            # 查找下一个页码标记的位置
            if i < len(page_breaks) - 1:
                next_marker = f"\n[PAGE_BREAK_{page_breaks[i+1]}]\n"
                end_pos = text.find(next_marker)
                if end_pos == -1:
                    # 如果找不到下一个标记，使用文本结尾
                    page_content = text[start_pos + len(marker):]
                else:
                    page_content = text[start_pos + len(marker):end_pos]
            else:
                # 最后一页
                page_content = text[start_pos + len(marker):]
                
            # 合并行并存储页面内容
            page_contents[page_num] = self._merge_lines(page_content)
        
        return page_contents

    def _process_page_content(self, page_content: str, page_num: int, 
                            title_counter: Dict[int, int], 
                            current_titles: List[List[str]], 
                            title_hierarchy: Dict) -> Tuple[List[Dict], Dict]:
        """处理单个页面的内容，提取标题和分块"""
        page_chunks = []
        page_titles = {}
        
        # 分割段落
        paragraphs = self._split_into_paragraphs(page_content, max_paragraph_length=self.chunk_size, use_spacy=False)
        
        # 处理段落
        for para in paragraphs:
            if not para.strip():
                continue
                
            # 检查是否是标题
            if self._is_title(para):
                title_info = self._process_title(
                    para, page_num, title_counter, current_titles, title_hierarchy
                )
                
                title_key = title_info['title_key']
                page_titles[title_key] = title_info['title_data']
                
                # 标题作为单独的块
                page_chunks.append({
                    'text': para.strip(),
                    'page': page_num,
                    'title_key': title_key
                })
            else:
                # 处理内容段落
                content_chunks = self._process_content_paragraph(
                    para, page_num, title_counter[page_num], title_hierarchy
                )
                page_chunks.extend(content_chunks)
        
        return page_chunks, page_titles

    def _process_title(self, title_text: str, page_num: int, 
                    title_counter: Dict[int, int], 
                    current_titles: List[List[str]], 
                    title_hierarchy: Dict) -> Dict:
        """处理标题文本，更新标题层次结构"""
        level = self._estimate_title_level(title_text)
        logger.info(f"识别到标题: {title_text} (页码: {page_num}) 级别: {level}")
        
        # 清除当前级别之后的所有标题
        for l in range(level, 10):
            current_titles[l] = []
        
        # 在当前级别添加新标题
        current_titles[level-1] = [title_text]
        
        # 构建完整标题路径 - 只包含有效级别的标题
        full_path_parts = []
        for l in range(10):
            if current_titles[l]:
                full_path_parts.extend(current_titles[l])
        
        full_title_path = " > ".join(full_path_parts)
        
        # 增加标题计数并创建唯一键
        title_counter[page_num] += 1
        title_key = f"{page_num}_{title_counter[page_num]}"
        
        # 创建标题数据
        title_data = {
            'title': title_text,
            'full_path': full_title_path,
            'level': level,
            'page': page_num
        }
        
        return {
            'title_key': title_key,
            'title_data': title_data
        }

    def _process_content_paragraph(self, para: str, page_num: int, 
                                max_title_count: int, 
                                title_hierarchy: Dict) -> List[Dict]:
        """处理内容段落，根据长度进行分块"""
        content_chunks = []
        
        if len(para) > self.chunk_size:
            # 长段落按句子拆分
            content_chunks = self._split_long_paragraph(
                para, page_num, max_title_count, title_hierarchy
            )
        else:
            # 段落长度未超过 chunk_size，直接作为一个块
            current_title_key = self._find_closest_title_key(title_hierarchy, page_num, max_title_count)
            content_chunks.append({
                'text': para.strip(),
                'page': page_num,
                'title_key': current_title_key
            })
        
        return content_chunks

    def _split_long_paragraph(self, para: str, page_num: int, 
                            max_title_count: int, 
                            title_hierarchy: Dict) -> List[Dict]:
        """将长段落按句子分割成多个块"""
        chunks = []
        sentences = re.split(r'(?<=[。！？\.!?])', para)
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # 找到当前页面最近的标题键
                current_title_key = self._find_closest_title_key(title_hierarchy, page_num, max_title_count)
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num,
                    'title_key': current_title_key
                })
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                
        # 添加最后一个块
        if current_chunk:
            current_title_key = self._find_closest_title_key(title_hierarchy, page_num, max_title_count)
            chunks.append({
                'text': current_chunk.strip(),
                'page': page_num,
                'title_key': current_title_key
            })
        
        return chunks

    def _find_closest_title_key(self, title_hierarchy, page_num, max_count):
        """找到当前页面最近的标题键"""
        # 从当前页面的最后一个标题开始向前查找
        for i in range(max_count, 0, -1):
            title_key = f"{page_num}_{i}"
            if title_key in title_hierarchy:
                return title_key
        
        # 如果当前页面没有标题，查找前面页面的标题
        # 优化：按页码筛选标题键，避免不必要的循环
        prev_page_keys = {}
        for key in title_hierarchy.keys():
            if '_' in key:  # 确保键的格式正确
                try:
                    key_page = int(key.split('_')[0])
                    if key_page < page_num:  # 只考虑当前页之前的页
                        if key_page not in prev_page_keys:
                            prev_page_keys[key_page] = []
                        prev_page_keys[key_page].append(key)
                except (ValueError, IndexError):
                    continue
        
        # 从最近的前一页开始查找
        for prev_page in range(page_num-1, 0, -1):
            if prev_page in prev_page_keys and prev_page_keys[prev_page]:
                # 按标题编号降序排序，获取最后一个标题
                sorted_keys = sorted(prev_page_keys[prev_page], 
                                    key=lambda x: int(x.split('_')[1]), 
                                    reverse=True)
                return sorted_keys[0]
        
        # 如果没有找到任何标题，返回一个默认键
        return "default"
    
    def _merge_lines(self, text: str) -> str:
        """合并连续的行，处理换行符，但保留标题的独立性"""
        lines = text.split('\n')
        merged_text = ""
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                merged_text += "\n"
                continue
                
            # 检查当前行是否是标题
            is_current_title = self._is_title(line)
            
            # 检查下一行是否是标题（如果有下一行）
            next_is_title = False
            if i < len(lines) - 1 and lines[i+1].strip():
                next_is_title = self._is_title(lines[i+1].strip())
            
            # 如果当前行是标题或者下一行是标题，保持独立行
            if is_current_title or next_is_title:
                if merged_text and not merged_text.endswith("\n"):
                    merged_text += "\n"
                merged_text += line + "\n"
            else:
                # 非标题行的正常合并逻辑
                if merged_text.endswith("\n") or not merged_text:
                    merged_text += line
                else:
                    merged_text += f" {line}"
        
        return merged_text
        
    def _split_into_paragraphs(self, text: str, max_paragraph_length: int = 300, use_spacy: bool = True) -> List[str]:
        """识别标题，合并非标题部分，并按语义或换行符分割文本"""
        paragraphs = []
        current_para = ""
        
        # 按行处理文本
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                # 跳过空行
                continue

            # 检查是否是标题
            if self._is_title(line):
                # 如果当前段落有内容，保存段落并清空
                if current_para:
                    paragraphs.append(current_para.strip())
                    current_para = ""
                # 标题作为单独的段落
                paragraphs.append(line.strip())
                continue

            # 非标题部分，合并行
            current_para = f"{current_para} {line}" if current_para else line

        # 添加最后一个段落
        if current_para:
            paragraphs.append(current_para.strip())

        # 如果使用 spaCy 进行分段
        if use_spacy:
            merged_text = " ".join(paragraphs)
            return self.splitter.merge_and_split_paragraphs([merged_text], max_paragraph_length)

        # 按段落长度分割
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > max_paragraph_length:
                sentences = re.split(r'(?<=[。！？\.!?])', para)  # 按标点符号拆分
                current_chunk = ""
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    if len(current_chunk) + len(sentence) > max_paragraph_length:
                        final_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                if current_chunk:
                    final_paragraphs.append(current_chunk.strip())
            else:
                final_paragraphs.append(para)

        return final_paragraphs
    
    def _extract_title_hierarchy(self, doc) -> Dict:
        """提取文档的标题层次结构"""
        title_hierarchy = {}
        current_titles = []  # 存储当前活动的标题层次
        
        # 提取目录结构
        toc = doc.get_toc()
        if toc:
            # 转换PyMuPDF的目录格式为更易处理的格式
            # toc格式: [[level, title, page, ...], ...]
            for item in toc:
                level, title, page = item[0], item[1], item[2]
                
                # 调整页码（PyMuPDF页码从1开始，我们需要从0开始）
                page = page - 1 if page > 0 else 0
                
                # 更新当前标题层次
                while len(current_titles) >= level:
                    current_titles.pop()
                current_titles.append(title)
                
                # 构建完整标题路径
                full_title_path = " > ".join(current_titles)
                title_hierarchy[page] = {
                    'title': title,
                    'full_path': full_title_path,
                    'level': level
                }
        
        # 如果没有目录，尝试从文本中提取标题
        if not title_hierarchy:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                lines = text.split('\n')
                
                for line in lines:
                    if self._is_title(line):
                        # 尝试确定标题级别
                        level = self._estimate_title_level(line)
                        
                        # 更新当前标题层次
                        while len(current_titles) >= level:
                            if current_titles:
                                current_titles.pop()
                        current_titles.append(line)
                        
                        # 构建完整标题路径
                        full_title_path = " > ".join(current_titles)
                        title_hierarchy[page_num] = {
                            'title': line,
                            'full_path': full_title_path,
                            'level': level
                        }
                        break
        
        return title_hierarchy

    def _estimate_title_level(self, title: str) -> int:
        """估计标题的层级"""
        # 根据标题格式估计层级
        if re.match(r'^第[一二三四五六七八九十\d]+[编, 篇]', title):
            return 1
        elif re.match(r'^第[一二三四五六七八九十\d]+[章]', title):
            return 2
        elif re.match(r'^第[一二三四五六七八九十\d]+[节]', title):
            return 3
        elif re.match(r'^[一二三四五六七八九十]+[、.]', title):
            return 4
        elif re.match(r'^[\d]+[、.]', title):
            return 5
        elif re.match(r'^（[一二三四五六七八九十]+）', title):
            return 6  
        elif re.match(r'^（[\d]+）', title):
            return 7
        elif re.match(r'^\([一二三四五六七八九十]+\)', title):
            return 8
        elif re.match(r'^\([\d]+\)', title):
            return 9
        else:
            return 1 
        
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 替换 Windows 风格换行符为标准换行符
        text = text.replace('\r', '\n')
        
        # 处理连字符断行
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # 去除单独的页码行，例如数字或 "第x页"
        text = re.sub(r'\n\s*(第\s*\d+\s*页)\s*\n', '\n\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
        
        # 去除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 调用过滤水印内容的方法
        return self._filter_watermark(text)

    def _is_title(self, text: str) -> bool:
        """判断文本是否为标题"""
        text = self._filter_watermark(text)
        
        # 如果文本不包含中文字符，则不是标题
        if not re.search(r'[\u4e00-\u9fff]', text):
            return False
        
        # 标题模式匹配
        title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇编]',
            r'^[一二三四五六七八九十\d]+[、\.]',
            r'^\([一二三四五六七八九十\d]+\)',
            r'^（[一二三四五六七八九十\d]+）'
        ]
        
        # 如果匹配任何标题模式，则是标题
        if any(re.match(pattern, text) for pattern in title_patterns) and len(text) < 20:
            return True
        
        # 对于短文本，只有在符合特定条件时才认为是标题
        # 例如：不包含标点符号、不以标点结尾等
        if len(text) < 20:
            # 排除明显是正文片段的短句
            if re.search(r'[。，；：？！…]', text):  # 包含这些标点的短句可能是正文片段
                return False
            # 排除引用或引文的片段
            if re.search(r'["\'』」》）]', text):
                return False
            # 排除以标点结尾的片段
            if re.search(r'[。，；：？！…]$', text):
                return False
            return True
            
        return False

    def _filter_watermark(self, text: str) -> str:
        """过滤文本中的水印内容"""
        watermark_keywords = [
            r'confidential', r'机密', r'保密文件', r'请勿外传', r'watermark',
            r'版权所有', r'all rights reserved', r'保留所有权利',r'严禁二改二传', r'改二传'
        ]
        lines = text.split('\n')
        return '\n'.join(line for line in lines if not any(re.search(keyword, line.lower()) for keyword in watermark_keywords))
    
    def _is_non_orthogonal_rotation(self, bbox):
        """
        判断边界框是否有非正交旋转（非90度或其倍数的旋转）
        
        参数:
            bbox: 包含四个值的元组或列表 (x0, y0, x1, y1)，表示边界框的左上角和右下角坐标
            
        返回:
            bool: True表示有非正交旋转，False表示没有或只有90度的旋转
        """
        # 提取边界框坐标
        x0, y0, x1, y1 = bbox
        
        # 计算宽度和高度
        width = x1 - x0
        height = y1 - y0
        
        # 计算宽高比
        aspect_ratio = width / height if height != 0 else float('inf')
        
        # 判断宽高比是否接近1:1（正方形通常表示45度旋转）
        # 可以根据需要调整阈值，这里使用0.9-1.1的范围
        if 0.9 <= aspect_ratio <= 1.1:
            return True
        
        # 如果宽高比不接近1，但也不是典型的水平或垂直文本，可能有轻微旋转
        # 典型的水平文本宽高比通常大于3，垂直文本宽高比通常小于0.33
        if 0.33 < aspect_ratio < 3.0 and abs(aspect_ratio - 1.0) > 0.1:
            return True
        
        # 否则认为是正常的水平或垂直文本（正交旋转或无旋转）
        return False

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