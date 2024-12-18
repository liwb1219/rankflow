# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from abc import ABC, abstractmethod
from typing import Literal, Union, Generator, Any, List, Optional
from pathlib import Path


class DataProcessor(ABC):
    """
    数据处理抽象基类, 提供通用的数据处理接口

    该类定义了数据处理的三个核心步骤:
        1. 读取数据(read_data):
        2. 加工数据(process_data):
        3. 转换数据(transform_data):

    支持两种处理模式:
        batch: 批处理模式, 一次性读取所有数据并处理
        stream: 流处理模式, 逐条读取数据并处理
    """
    def __init__(self, mode: Literal['batch', 'stream'] = 'batch', encoding: str = 'utf-8'):
        if mode not in {'batch', 'stream'}:
            str_b = f'\033[1;33mbatch\033[0m'   # 黄色加粗
            str_s = f'\033[1;33mstream\033[0m'  # 黄色加粗
            raise ValueError(
                f'\033[1;31mThe mode parameter must be either {str_b} \033[1;31mor {str_s}\033[0m'
            )
        self.mode = mode
        self.encoding = encoding

    def read_data(self, file_path: Union[str, Path]) -> Generator[str, None, None]:
        """ 读取数据的方法, 具体实现取决于数据来源, 默认按行读取 """
        # 检查路径合法性
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            file_path = f'\033[1;32m{file_path}\033[0m'  # 绿色加粗
            raise FileNotFoundError(
                f'\033[1;31mNo such file or directory: {file_path}\033[0m'
            )
        with open(file_path, 'r', encoding=self.encoding) as file:
            for line in file:
                yield line

    @abstractmethod
    def process_data(self, data: Any) -> Any:
        """ 加工数据的抽象方法, 用于对原始数据进行初步加工, 每个子类必须覆盖此方法来定义具体的加工逻辑 """
        pass

    @abstractmethod
    def transform_data(self, data: Any) -> Any:
        """ 转换数据的抽象方法, 用于进一步转换加工后的数据, 每个子类必须覆盖此方法来定义具体的转换逻辑 """
        pass

    def run(
        self,
        file_path: Union[str, Path],
        mode: Optional[Literal['batch', 'stream']] = None,
        encoding: Optional[str] = None,
    ) -> Union[List[Any], Generator[Any, None, None]]:
        """
        根据处理模式选择运行批处理或流处理, 并允许覆盖默认处理模式和文件编码
        :param file_path: 文件路径
        :param mode: 处理模式, 可选值为 'batch' 或 'stream', 如果为 None, 则使用类的默认处理模式
        :param encoding: 文件编码, 如果为 None, 则使用类的默认文件编码
        :return: 返回处理后的数据, 批处理模式返回列表, 流处理模式返回生成器
        """
        self.mode = mode if mode is not None else self.mode
        self.encoding = encoding if encoding is not None else self.encoding

        if self.mode == 'batch':
            return self._run_batch(file_path)
        elif self.mode == 'stream':
            return self._run_stream(file_path)
        else:
            raise RuntimeError(
                f'\033[1;31mInvalid mode: \033[1;33m{self.mode}\033[1;31m. '
                f'Please choose from \033[1;33m{['batch', 'stream']}\033[0m'
            )

    def _run_batch(self, file_path: Union[str, Path]) -> List[Any]:
        data_list = []
        for data in self.read_data(file_path):
            data = self.process_data(data)
            data = self.transform_data(data)
            data_list.append(data)
        return data_list

    def _run_stream(self, file_path: Union[str, Path]) -> Generator[Any, None, None]:
        for data in self.read_data(file_path):
            data = self.process_data(data)
            data = self.transform_data(data)
            yield data
