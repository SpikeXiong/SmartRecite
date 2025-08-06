import time
from flask import Blueprint, request, jsonify, current_app
import os
import logging
from werkzeug.utils import secure_filename
import tempfile
from textCodes.faiss.pdf_importer import PDFImporter

# 创建蓝图
knowledge_bp = Blueprint('knowledge', __name__)

# 配置日志
logger = logging.getLogger(__name__)

# 允许的文件类型
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@knowledge_bp.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """处理PDF上传并导入到知识库"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '不支持的文件类型，仅支持PDF'}), 400
    
    try:
        # 获取参数
        password = request.form.get('password', None)
        if password and password.strip() == '':
            password = None
            
        metadata = {}
        title = request.form.get('title', None)
        author = request.form.get('author', None)
        category = request.form.get('category', None)
        
        if title:
            metadata['title'] = title
        if author:
            metadata['author'] = author
        if category:
            metadata['category'] = category
        
        # 保存到临时文件
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # 获取知识库实例
        kb = current_app.config.get('KNOWLEDGE_BASE')
        if not kb:
            return jsonify({'success': False, 'message': '知识库未初始化'}), 500
        
        # 创建PDF导入器
        importer = PDFImporter(kb, chunk_size=500, overlap=50)
        try:
            # 导入PDF
            imported_count, errors = importer.import_pdf(temp_path, password, metadata)

             # 等待一小段时间确保文件被完全释放
            time.sleep(0.5)
            
            # 尝试删除临时文件，但不要因为删除失败而中断处理
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"无法删除临时文件 {temp_path}: {str(e)}")
                # 这里我们只记录警告，不中断流程
                
        except Exception as e:
            logger.exception("PDF导入失败")
            # 尝试清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            return jsonify({'success': False, 'message': f'处理失败: {str(e)}'}), 500
        
        if imported_count > 0:
            result = {
                'success': True,
                'message': f'成功导入 {imported_count} 个文本块',
                'imported_count': imported_count,
                'filename': filename
            }
            
            if errors:
                result['warnings'] = errors
                
            return jsonify(result), 200
        else:
            return jsonify({
                'success': False,
                'message': '导入失败，未能提取任何文本',
                'errors': errors
            }), 400
            
    except Exception as e:
        logger.exception("PDF导入失败")
        return jsonify({'success': False, 'message': f'处理失败: {str(e)}'}), 500

@knowledge_bp.route('/knowledge_stats', methods=['GET'])
def knowledge_stats():
    """获取知识库统计信息"""
    kb = current_app.config.get('KNOWLEDGE_BASE')
    if not kb:
        return jsonify({'success': False, 'message': '知识库未初始化'}), 500
    
    try:
        # 获取基本统计信息
        stats = kb.get_stats()
        
        # 添加文档列表信息（如果知识库实现支持）
        if hasattr(kb, 'get_documents') and callable(getattr(kb, 'get_documents')):
            stats['documents'] = kb.get_documents()
        else:
            # 如果知识库没有实现get_documents方法，提供一个简单的模拟数据
            # 在实际应用中，你应该扩展知识库类以提供这个功能
            stats['documents'] = []
        
        return jsonify({'success': True, 'stats': stats}), 200
    except Exception as e:
        logger.exception("获取知识库统计信息失败")
        return jsonify({'success': False, 'message': f'获取统计信息失败: {str(e)}'}), 500

@knowledge_bp.route('/documents', methods=['GET'])
def get_documents():
    """获取已导入的文档列表"""
    kb = current_app.config.get('KNOWLEDGE_BASE')
    if not kb:
        return jsonify({'success': False, 'message': '知识库未初始化'}), 500
    
    try:
        # 如果知识库实现了get_documents方法
        if hasattr(kb, 'get_documents') and callable(getattr(kb, 'get_documents')):
            documents = kb.get_documents()
            return jsonify({'success': True, 'documents': documents}), 200
        else:
            # 返回空列表
            return jsonify({'success': True, 'documents': []}), 200
    except Exception as e:
        logger.exception("获取文档列表失败")
        return jsonify({'success': False, 'message': f'获取文档列表失败: {str(e)}'}), 500