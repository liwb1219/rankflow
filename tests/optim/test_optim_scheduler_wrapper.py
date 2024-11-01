# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from rankflow.optim import OptimSchedulerWrapper


class ToyModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


class ToyModelV2(nn.Module):
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
        model = ToyModelV1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        loss_fct = nn.MSELoss()
        for data, label in zip(torch.randn(100, 1), torch.randn(100, 1)):
            loss = loss_fct(model(data), label)

            """
            PyTorch模型训练核心步骤:
                1. 梯度清零: optimizer.zero_grad()
                2. 反向传播: loss.backward()
                3. 参数更新: optimizer.step()
            
            它们之间的顺序要求:
                1. 梯度清零 只能写在最前面或最后面
                2. 反向传播 要写在 参数更新 之前
            
            所以有以下两种写法:
            (1) optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            (2) loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            """

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播求解梯度
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()  # 更新权重参数
            scheduler.step()  # 更新学习率

            print(scheduler.get_lr(), {name: param for name, param in model.named_parameters()})



if __name__ == '__main__':
    unittest.main()
