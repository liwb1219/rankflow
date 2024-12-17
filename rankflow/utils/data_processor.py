# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from abc import ABC, abstractmethod
from typing import Literal, Union, Generator, Any
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
            str_b = f"\033[1;33mbatch\033[0m"   # 黄色加粗
            str_s = f"\033[1;33mstream\033[0m"  # 黄色加粗
            raise ValueError(
                f"\033[1;31mThe mode parameter must be either {str_b} \033[1;31mor {str_s}\033[0m"
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

    def run(self):
        pass

    def _run_batch(self):
        pass

    def _run_stream(self):
        pass


class DataProcessor1(ABC):
    def run(self) -> None:
        """
        运行整个数据处理流程。依据实例化的模式（批处理或流处理）选择运行的方式。
        """
        print("Starting data processing...")
        try:
            self.read_data()
            self.process_data()
            self.transform_data()
            print("Data processing completed successfully.")
        except Exception as e:
            print(f"An error occurred during data processing: {e}")

    def _run_batch(self) -> None:
        """
        执行批处理模式下的数据处理流程。这个方法可以在继承类中被覆写以适应不同的需求。
        默认实现为空。
        """
        pass

    def _run_stream(self) -> None:
        """
        执行流处理模式下的数据处理流程。这个方法可以在继承类中被覆写以适应不同的需求。
        默认实现为空。
        """
        pass


class DataProcessor2(ABC):
    @abstractmethod
    def transform_data(self, processed_data: Any) -> Any:
        """
        转换数据的抽象方法，用于进一步转换加工后的数据。

        :param processed_data: 加工后的数据项。
        :return: 转换后的数据项。
        """
        pass

    def run(self):
        """
        根据模式运行数据处理流程。

        根据 self.mode 的值，选择批处理模式或流处理模式。
        """
        if self.mode == 'batch':
            self._run_batch()
        elif self.mode == 'stream':
            self._run_stream()

    def _run_batch(self):
        """
        批处理模式：一次性读取所有数据并处理。

        1. 调用 read_data() 读取所有数据。
        2. 调用 process_data() 对每条数据进行加工。
        3. 调用 transform_data() 对加工后的数据进行转换。
        4. 调用 _write_data() 输出转换后的数据。
        """
        data_list: list[Any] = list(self.read_data())  # 读取所有数据
        processed_data_list: list[Any] = [self.process_data(data) for data in data_list]  # 加工数据
        transformed_data_list: list[Any] = [self.transform_data(processed_data) for processed_data in processed_data_list]  # 转换数据
        self._write_data(transformed_data_list)  # 输出数据

    def _run_stream(self):
        """
        流处理模式：逐条读取数据并处理。

        1. 调用 read_data() 逐条读取数据。
        2. 调用 process_data() 对每条数据进行加工。
        3. 调用 transform_data() 对加工后的数据进行转换。
        4. 调用 _write_data() 输出转换后的数据。
        """
        for data in self.read_data():
            processed_data: Any = self.process_data(data)  # 加工数据
            transformed_data: Any = self.transform_data(processed_data)  # 转换数据
            self._write_data(transformed_data)  # 输出数据

    def _write_data(self, transformed_data: Any):
        """
        输出处理后的数据（默认打印）。

        :param transformed_data: 转换后的数据项。
        """
        print(transformed_data)


if __name__ == '__main__':
    processor = DataProcessor()


