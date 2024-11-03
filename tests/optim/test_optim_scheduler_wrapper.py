# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import unittest
import random
from typing import Tuple
from transformers import get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rankflow.optim import OptimSchedulerWrapper


# 递归函数来铺平嵌套列表
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def build_optimizer_and_scheduler(
        model: nn.Module,
        num_training_steps: int,
        num_warmup_steps: int,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-5,
        scheduler_type: str = 'linear',
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


class ToyDataset(Dataset):
    def __init__(self, data_size: int = 100):
        self.data = [(random.random(), random.random()) for _ in range(data_size)]
        self.label = [random.random() for _ in range(data_size)]

    def __getitem__(self, item):
        return torch.tensor(self.data[item]), torch.tensor(self.label[item])

    def __len__(self):
        return len(self.data)


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


class ToyModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class TestOptimSchedulerWrapper(unittest.TestCase):

    # 测试基础功能是否和常用模版一致
    def test_optim_scheduler_wrapper_1(self):
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

    # 测试epoch训练(无梯度累积)
    def test_optim_scheduler_wrapper_2(self):
        max_epochs = 3
        batch_size = 8
        warmup_ratio = 0.1

        dataset = ToyDataset(data_size=4000)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        num_training_steps = max_epochs * len(data_loader)
        num_warmup_steps = int(warmup_ratio * num_training_steps)

        gradient_clipping_max_norm = 1.0

        loss_fct = nn.MSELoss()

        model_a = ToyModelV3()
        model_b = ToyModelV3()

        # 将模型a的参数复制给模型b
        model_b.load_state_dict(model_a.state_dict())

        optimizer_a, scheduler_a = build_optimizer_and_scheduler(
            model=model_a,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        res_a_list = []
        for epoch in range(max_epochs):
            for data, label in data_loader:
                loss = loss_fct(model_a(data), label)

                loss.backward()  # 反向传播求解梯度
                nn.utils.clip_grad_norm_(model_a.parameters(), gradient_clipping_max_norm)  # 梯度裁剪
                optimizer_a.step()  # 更新权重参数
                optimizer_a.zero_grad()  # 梯度清零

                scheduler_a.step()  # 更新学习率

                res_a_list.append(
                    [
                        loss.item(),
                        scheduler_a.get_lr(),
                        [param.cpu().tolist() for param in model_a.parameters()],
                    ]
                )

        optimizer_b, scheduler_b = build_optimizer_and_scheduler(
            model=model_b,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )
        optim_scheduler_b = OptimSchedulerWrapper(
            optimizer_b,
            scheduler_b,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            gradient_accumulation_steps=1,
            enable_amp=False,
            num_training_steps=num_training_steps,
        )

        res_b_list = []
        for epoch in range(max_epochs):
            for data, label in data_loader:
                loss = loss_fct(model_b(data), label)

                optim_scheduler_b.update_params(loss)
                optim_scheduler_b.update_lr()

                res_b_list.append(
                    [
                        loss.item(),
                        optim_scheduler_b.get_lr(),
                        [param.cpu().tolist() for param in model_b.parameters()],
                    ]
                )

        res_a_list = flatten_list(res_a_list)
        res_b_list = flatten_list(res_b_list)
        cnt_a = 0
        cnt_b = 0
        for x, y in zip(res_a_list, res_b_list):
            if abs(x - y) < 0.001:
                cnt_a += 1
            else:
                cnt_b += 1
                print(x, y)
        print(cnt_a, cnt_b)


        # self.assertListEqual(res_a_list, res_b_list)

    # 测试epoch训练(有梯度累积)
    def test_optim_scheduler_wrapper_3(self):
        pass


if __name__ == '__main__':
    unittest.main()
