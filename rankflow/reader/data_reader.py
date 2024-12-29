# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import torch
from pathlib import Path
from typing import Union
from utils import BaseDataProcessor
from torch.utils.data import Dataset, IterableDataset


class MapDataReader(Dataset):
    def __init__(self, data_path: Union[str, Path]):
        data_processor = BaseDataProcessor()
        self.data = data_processor.run(data_path, 'batch')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class IterableDataReader(IterableDataset):
    def __init__(self, data_path: Union[str, Path], rank: int = 0, world_size: int = 1):
        self.data_processor = BaseDataProcessor()
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.train_data_path = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            worker_num = 1
        else:
            worker_id = worker_info.id
            worker_num = worker_info.num_workers

        for idx, data in enumerate(self.data_processor.run(self.data_path, 'stream')):
            if idx % (self.world_size * worker_num) == self.rank * worker_num + worker_id:
                yield data
