# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from abc import ABC, abstractmethod
from typing import Literal


class DataProcessor(ABC):
    """ 数据处理抽象基类, 提供通用的数据处理接口 """
    def __init__(self, mode: Literal['batch', 'stream'] = 'batch', encoding: str = 'utf-8'):
        if mode not in {'batch', 'stream'}:
            str_b = f"\033[1;33mbatch\033[0m"   # 黄色加粗
            str_s = f"\033[1;33mstream\033[0m"  # 黄色加粗
            raise ValueError(
                f"\033[1;31mThe mode parameter must be either {str_b} \033[1;31mor {str_s}\033[0m"
            )
        self.mode = mode
        self.encoding = encoding

    def read_data(self):
        pass

    def process_data(self):
        pass

    @abstractmethod
    def format_output(self, processed_data: Any) -> Any:
        """
        抽象方法：将加工后的数据转换为指定格式
        :param processed_data: 加工后的数据
        :return: 格式化后的数据
        """
        pass

    def run(self) -> Any:
        """
        运行数据处理流程
        :return: 最终的输出结果
        """

    @abstractmethod
    def format_data(self, data: pd.DataFrame, output_format: str) -> Union[str, dict, pd.DataFrame]:
        """ 返回指定格式的数据 """
        pass

if __name__ == '__main__':
    processor = DataProcessor()


