# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import argparse
import yaml
import json
from pathlib import Path


class RFArgumentParser(argparse.ArgumentParser):
    """
    参数优先级: 命令行 > 配置文件 > 默认
    内部解析器: 只用于解析配置文件参数,并将其余参数传递给外部解析器.
    外部解析器: 负责解析所有通过add_argument方法添加的参数.
    例如:
    parser = RFArgumentParser()
    parser.add_argument('--name', type=str, default='liwenbiao')
    parser.add_argument('--age', type=int, default=35)
    args = parser.parse_args()
    python rf_argparser.py --config=config.yaml --name=lwb --age=18
    会将config.yaml传给内部解析器, --name=lwb --age=18传给外部解析器
    """
    def __init__(self, *args, **kwargs):
        self.dest_set = set()
        super().__init__(*args, **kwargs)
        self.parser = argparse.ArgumentParser(add_help=False)  # 创建一个解析对象
        self.parser.add_argument('--config', type=str, default=None, help='JSON or YAML format.')

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)  # 调用父类的add_argument方法返回一个action对象
        self.dest_set.add(action.dest)  # 收集参数用于后续校验
        return action

    def parse_args(self, args=None):
        args, argv = self.parser.parse_known_args(args)
        if args.config is not None:
            default_args = self._load_config_file(args.config)
            self._validate_default_args(default_args, args.config)
            self.set_defaults(**default_args)

        return super().parse_args(argv)

    def _validate_default_args(self, default_args: dict, config_file: str):
        for arg in default_args:
            if arg not in self.dest_set:
                arg = f'\033[1;33m"{arg}"\033[0m'                   # 黄色加粗
                config_file = f'\033[1;32m"{config_file}"\033[0m'   # 绿色加粗
                raise ValueError(
                    f'\033[1;31mInvalid parameter(s) {arg} \033[1;31mfound in configuration file: {config_file}\033[0m'
                )

    @staticmethod
    def _load_config_file(file_path: str) -> dict:
        """
        :param file_path: 配置文件的路径
        :return: 参数字典
        """
        # 检查路径合法性
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            file_path = f'\033[1;32m"{file_path}"\033[0m'  # 绿色加粗
            raise FileNotFoundError(
                f'\033[1;31mNo such file or directory: {file_path}\033[0m'
            )

        # 获取文件后缀(扩展名)
        file_extension = path.suffix.lower()

        # 根据后缀加载相应格式的文件
        if file_extension == '.json':
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        elif file_extension in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            file_extension = f'\033[1;33m"{file_extension}"\033[0m'     # 黄色加粗
            supported_formats = '\033[1;32m.json, .yaml, .yml\033[0m'   # 绿色加粗
            raise ValueError(
                f'\033[1;31mUnsupported file format: {file_extension}\033[1;31m. '
                f'Supported formats are: {supported_formats}\033[0m'
            )


if __name__ == '__main__':
    parser = RFArgumentParser()
    parser.add_argument('--name', type=str, default='liwenbiao')
    parser.add_argument('--gender', type=str, default='male')
    parser.add_argument('--age', type=int, default=35)
    parser.add_argument('--job', type=str, default='dogsbody')
    print(parser.parse_args())
