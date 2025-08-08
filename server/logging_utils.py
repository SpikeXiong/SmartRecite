import logging
import os

def setup_logger(name=None, log_file='app.log', level=logging.INFO):
    """
    设置日志记录器，同时输出到控制台和文件
    
    Args:
        name: 日志记录器名称，默认为None（使用root logger）
        log_file: 日志文件名
        level: 日志级别
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 获取logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的handlers
    if logger.handlers:
        logger.handlers.clear()

    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建文件handler
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger