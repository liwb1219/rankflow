# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import json
import torch
from rankflow.utils.data_processor import DataProcessor

class BaseDataProcessor(DataProcessor):
    def process_data(self, data):
        data = json.loads(data)
        return data

    def transform_data(self, data):
        return {
            'input_ids': torch.tensor(data['input_ids']),
            'attention_mask': torch.tensor(data['attention_mask']),
            'token_type_ids': torch.tensor(data['token_type_ids']),
            'labels': torch.tensor(data['labels']),
        }
