# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
from rankflow import RFArgumentParser

class TestRFArgumentParser(unittest.TestCase):

    # 测试默认参数
    def test_parser(self):
        parser = RFArgumentParser()
        parser.add_argument('--name', type=str, default='liwenbiao')
        parser.add_argument('--gender', type=str, default='male')
        parser.add_argument('--age', type=int, default=35)
        parser.add_argument('--job', type=str, default='dogsbody')
        parser.add_argument('--email', type=str, default='1758123337@qq.com')
        args = parser.parse_args([])
        self.assertEqual(args.name, 'liwenbiao')
        self.assertEqual(args.gender, 'male')
        self.assertEqual(args.age, 35)
        self.assertEqual(args.job, 'dogsbody')
        self.assertEqual(args.email, '1758123337@qq.com')




if __name__ == '__main__':
    unittest.main()
