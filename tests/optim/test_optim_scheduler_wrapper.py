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
        input_tensors = torch.randn(100, 1)
        label_tensors = torch.randn(100, 1)
        learning_rate = 0.1
        step_size = 1
        gamma = 0.99
        gradient_clipping_max_norm = 1.0
        loss_fct = nn.MSELoss()

        model_a = ToyModelV2()
        model_b = ToyModelV2()

        # 将模型a的参数复制给模型b
        model_b.load_state_dict(model_a.state_dict())

        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=learning_rate)
        scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, step_size=step_size, gamma=gamma)
        res_a_list = []
        for data, label in zip(input_tensors, label_tensors):
            loss = loss_fct(model_a(data), label)

            loss.backward()  # 反向传播求解梯度
            nn.utils.clip_grad_norm_(model_a.parameters(), gradient_clipping_max_norm)  # 梯度裁剪
            optimizer_a.step()  # 更新权重参数
            optimizer_a.zero_grad()  # 梯度清零

            scheduler_a.step()  # 更新学习率

            res_a_list.append(
                (
                    loss.item(),
                    scheduler_a.get_lr(),
                    [param.cpu().tolist() for param in model_a.parameters()],
                )
            )

        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=learning_rate)
        scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=step_size, gamma=gamma)
        optim_scheduler_b = OptimSchedulerWrapper(
            optimizer_b,
            scheduler_b,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            gradient_accumulation_steps=1,
            enable_amp=False,
            num_training_steps=-1,
        )
        res_b_list = []
        for data, label in zip(input_tensors, label_tensors):
            loss = loss_fct(model_b(data), label)

            optim_scheduler_b.update_params(loss)
            optim_scheduler_b.update_lr()

            res_b_list.append(
                (
                    loss.item(),
                    optim_scheduler_b.get_lr(),
                    [param.cpu().tolist() for param in model_b.parameters()],
                )
            )

        self.assertListEqual(res_a_list, res_b_list)



if __name__ == '__main__':
    unittest.main()
