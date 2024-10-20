# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import logging
from pathlib import Path
from typing import Union, Optional


class ColoredFormatter(logging.Formatter):
    """
    显示彩色文本方式: `开头` + `文本` + `结尾`
    开头: \033[显示方式;前景色;背景色m
    结尾: \033[0m

    显示方式: 0(默认值), 1(高亮), 22(非粗体), 4(下划线), 24(非下划线), 5(闪烁), 25(非闪烁), 7(反显), 27(非反显)
    前景色: 30(黑色), 31(红色), 32(绿色), 33(黄色), 34(蓝色), 35(紫色), 36(青色), 37(白色)
    背景色: 40(黑色), 41(红色), 42(绿色), 43(黄色), 44(蓝色), 45(紫色), 46(青色), 47(白色)

    常见开头格式：
    \033[0m         默认
    \033[1;31m      高亮 红色字体
    \033[1;32;40m   高亮 绿色字体 黑色背景
    \033[0;31;46m   正常 红色字体 青色背景
    """

    RESET = '\033[0m'       # 重置所有属性(默认)
    BLACK = '\033[30m'      # 黑色文本
    RED = '\033[31m'        # 红色文本
    GREEN = '\033[32m'      # 绿色文本
    YELLOW = '\033[33m'     # 黄色文本
    BLUE = '\033[34m'       # 蓝色文本
    MAGENTA = '\033[35m'    # 紫色文本
    CYAN = '\033[36m'       # 青色文本
    WHITE = '\033[37m'      # 白色文本

    HIGHLIGHT_BLACK = '\033[1;30m'      # 高亮黑色文本
    HIGHLIGHT_RED = '\033[1;31m'        # 高亮红色文本
    HIGHLIGHT_GREEN = '\033[1;32m'      # 高亮绿色文本
    HIGHLIGHT_YELLOW = '\033[1;33m'     # 高亮黄色文本
    HIGHLIGHT_BLUE = '\033[1;34m'       # 高亮蓝色文本
    HIGHLIGHT_MAGENTA = '\033[1;35m'    # 高亮紫色文本
    HIGHLIGHT_CYAN = '\033[1;36m'       # 高亮青色文本
    HIGHLIGHT_WHITE = '\033[1;37m'      # 高亮白色文本

    def __init__(self, enable_highlight_colors: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_start = {
            'DEBUG': self.HIGHLIGHT_BLUE if enable_highlight_colors else self.BLUE,
            'INFO': self.HIGHLIGHT_GREEN if enable_highlight_colors else self.GREEN,
            'WARNING': self.HIGHLIGHT_YELLOW if enable_highlight_colors else self.YELLOW,
            'ERROR': self.HIGHLIGHT_RED if enable_highlight_colors else self.RED,
            'CRITICAL': self.HIGHLIGHT_WHITE if enable_highlight_colors else self.WHITE,
        }
        self.format_end =self.RESET

    def format(self, record):
        log_start = self.format_start[record.levelname]
        log_end = self.format_end
        log_record = log_start + super().format(record) + log_end
        return log_record


def setup_logger(
    log_level: Union[int, str] = logging.DEBUG,
    log_file: Optional[str] = None,
    file_mode: Optional[str] = None,
    enable_highlight_colors: bool = False,
) -> logging.Logger:
    # 创建日志器并设置日志输出的最低等级(CRITICAL > ERROR > WARNING > INFO > DEBUG)
    logger = logging.getLogger(name='rankflow')
    logger.setLevel(log_level)

    # 清除现有处理器
    for handler in logger.handlers[:]:  # 使用[:]确保不会在遍历时修改列表
        logger.removeHandler(handler)
        handler.close()  # 关闭处理器 释放资源

    # 日志格式
    default_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'
    default_time_format = '%Y-%m-%d %H:%M:%S'

    # 创建控制台格式器
    stream_formatter = ColoredFormatter(
        fmt=default_format,
        datefmt=default_time_format,
        enable_highlight_colors=enable_highlight_colors,
    )
    # 创建控制台处理器并设置输出格式
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(stream_formatter)
    # 将控制台处理器添加到日志器中
    logger.addHandler(stream_handler)

    try:
        # 创建文件格式器
        file_formatter = logging.Formatter(fmt=default_format, datefmt=default_time_format)
        # 创建文件处理器并设置输出格式
        Path(log_file).resolve().parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        # 将文件处理器添加到日志器中
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f'Failed to create file handler for {log_file}: {e}')

    return logger


if __name__ == '__main__':
    # test_logger = setup_logger(log_level='DEBUG', log_file='worker.log', file_mode='a')
    test_logger = setup_logger()
    test_logger.info('rankflow project')
    test_logger.debug('rankflow project')
    test_logger.warning('rankflow project')
    test_logger.error('rankflow project')
    test_logger.critical('rankflow project')
