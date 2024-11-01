# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from rankflow.optim import OptimSchedulerWrapper


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class TestOptimSchedulerWrapper(unittest.TestCase):
    def test_optim_scheduler_wrapper(self):
        model = ToyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        optim_scheduler = OptimSchedulerWrapper(
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clipping_max_norm=1.0,
            gradient_accumulation_steps=1,
            enable_amp=True,
            num_training_steps=-1,
        )

        loss_fct = nn.MSELoss()
        for data, label in zip(torch.randn(100, 1), torch.randn(100, 1)):
            loss = loss_fct(model(data), label)
            optim_scheduler.update_params(loss)
            optim_scheduler.update_lr()



if __name__ == '__main__':
    unittest.main()
