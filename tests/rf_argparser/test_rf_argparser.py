# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
from rankflow import RFArgumentParser


class TestRFArgumentParser(unittest.TestCase):

    # 测试默认参数
    def test_case_1(self):
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

    # 测试json文件(优先级高于默认参数)
    def test_case_2(self):
        parser = RFArgumentParser()
        parser.add_argument('--name', type=str, default='liwenbiao')
        parser.add_argument('--gender', type=str, default='male')
        parser.add_argument('--age', type=int, default=35)
        parser.add_argument('--job', type=str, default='dogsbody')
        parser.add_argument('--email', type=str, default='1758123337@qq.com')
        """
        {
            "name": "lwb",
            "gender": "male",
            "age": 18,
            "job": "Algorithm Engineer",
            "email": "1758123337@qq.com"
        }
        """
        args = parser.parse_args([
            '--config', 'config.json',
        ])

        self.assertEqual(args.name, 'lwb')
        self.assertEqual(args.gender, 'male')
        self.assertEqual(args.age, 18)
        self.assertEqual(args.job, 'Algorithm Engineer')
        self.assertEqual(args.email, '1758123337@qq.com')

    # 测试yaml文件(优先级高于默认参数)
    def test_case_3(self):
        parser = RFArgumentParser()
        parser.add_argument('--name', type=str, default='liwenbiao')
        parser.add_argument('--gender', type=str, default='male')
        parser.add_argument('--age', type=int, default=35)
        parser.add_argument('--job', type=str, default='dogsbody')
        parser.add_argument('--email', type=str, default='1758123337@qq.com')
        """
        name: liwb
        gender: male
        age: 25
        job: Algorithm Expert
        email: liwenbiao@qq.com
        """
        args = parser.parse_args([
            '--config', 'config.yaml'
        ])

        self.assertEqual(args.name, 'liwb')
        self.assertEqual(args.gender, 'male')
        self.assertEqual(args.age, 25)
        self.assertEqual(args.job, 'Algorithm Expert')
        self.assertEqual(args.email, 'liwenbiao@qq.com')

    # 测试命令行(优先级最高)
    def test_case_4(self):
        parser = RFArgumentParser()
        parser.add_argument('--name', type=str, default='liwenbiao')
        parser.add_argument('--gender', type=str, default='male')
        parser.add_argument('--age', type=int, default=35)
        parser.add_argument('--job', type=str, default='dogsbody')
        parser.add_argument('--email', type=str, default='1758123337@qq.com')
        """
        {
            "name": "lwb",
            "gender": "male",
            "age": 18,
            "job": "Algorithm Engineer",
            "email": "1758123337@qq.com"
        }
        """
        args = parser.parse_args([
            '--config', 'config.json',
            '--name', 'liwb',
            '--age', '28',
            '--email', 'liwb@qq.com'
        ])

        self.assertEqual(args.name, 'liwb')
        self.assertEqual(args.gender, 'male')
        self.assertEqual(args.age, 28)
        self.assertEqual(args.job, 'Algorithm Engineer')
        self.assertEqual(args.email, 'liwb@qq.com')


if __name__ == '__main__':
    unittest.main()
