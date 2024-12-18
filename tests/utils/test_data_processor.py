# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
from rankflow.utils.data_processor import DataProcessor


class MyDataProcessor(DataProcessor):
    def process_data(self, data):
        return data

    def transform_data(self, data):
        return data


class TestLogger(unittest.TestCase):
    # 测试日志器
    def test_data_processor_1(self):
        pass


if __name__ == '__main__':
    unittest.main()
