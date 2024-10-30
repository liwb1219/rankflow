# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
from rankflow.utils.logger import setup_logger


class TestLogger(unittest.TestCase):
    # 测试日志器
    def test_setup_logger_1(self):
        logger = setup_logger(
            log_level='DEBUG',
            log_file=None,
            file_mode=None,
            enable_highlight_colors=False,
        )
        logger.debug('Test logger_1 debug')
        logger.info('Test logger_1 info')
        logger.warning('Test logger_1 warning')
        logger.error('Test logger_1 error')
        logger.critical('Test logger_1 critical')

    # 测试日志器(高亮字体)
    def test_setup_logger_2(self):
        logger = setup_logger(
            log_level='DEBUG',
            log_file=None,
            file_mode=None,
            enable_highlight_colors=True,
        )
        logger.debug('Test logger_2 debug')
        logger.info('Test logger_2 info')
        logger.warning('Test logger_2 warning')
        logger.error('Test logger_2 error')
        logger.critical('Test logger_2 critical')

    # 测试日志器并写入文件
    def test_setup_logger_3(self):
        logger = setup_logger(
            log_level='DEBUG',
            log_file='worker_3.log',
            file_mode='a',
            enable_highlight_colors=False,
        )
        logger.debug('Test logger_3 debug')
        logger.info('Test logger_3 info')
        logger.warning('Test logger_3 warning')
        logger.error('Test logger_3 error')
        logger.critical('Test logger_3 critical')

    # 测试日志器并写入文件(高亮字体)
    def test_setup_logger_4(self):
        logger = setup_logger(
            log_level='DEBUG',
            log_file='worker_4.log',
            file_mode='a',
            enable_highlight_colors=True,
        )
        logger.debug('Test logger_4 debug')
        logger.info('Test logger_4 info')
        logger.warning('Test logger_4 warning')
        logger.error('Test logger_4 error')
        logger.critical('Test logger_4 critical')


if __name__ == '__main__':
    unittest.main()
