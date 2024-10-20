# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import torch
from torch.cuda.amp import GradScaler


class OptimSchedulerWrapper:
    """
    包装优化器和学习率调度器的类, 支持梯度清零、反向传播、梯度裁剪、梯度累积、权重更新、学习率更新以及混合精度训练等功能

    使用示例:
    optim_scheduler = OptimSchedulerWrapper(
        optimizer,
        scheduler,
        gradient_clipping_max_norm=1.0,
        gradient_accumulation_steps=1,
        enable_amp=True
    )

    for epoch in range(epochs):
        for data in data_loader:
            loss = model(**data)['loss']
            optim_scheduler.update_params(loss)
            optim_scheduler.update_learning_rate()
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        gradient_clipping_max_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        enable_amp: bool = True,
        num_training_steps: int = -1,
    ):
        """
        :param optimizer: 优化器实例(如 torch.optim.Adam, torch.optim.SGD 等)
        :param scheduler: 学习率调度器实例(如 torch.optim.lr_scheduler.StepLR 等)
        :param gradient_clipping_max_norm: 梯度裁剪的最大阈值(默认L2范数), 默认值为1.0
        :param gradient_accumulation_steps: 梯度累积的步数, 默认为1, 即不进行累计
        :param enable_amp: 开启混合精度训练, 默认True
        :param num_training_steps: 总共训练步数, 默认为-1, 表示未指定
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_amp = enable_amp
        self.scaler = GradScaler() if enable_amp else None
        self._num_training_steps = num_training_steps  # 总共需要的训练步数
        self._current_step = 0  # 当前训练步数
        if self._num_training_steps != -1:
            # 计算最后一次更新参数梯度累计的步数
            remaining_steps = self._num_training_steps % self.gradient_accumulation_steps
            self._remaining_steps = remaining_steps if remaining_steps > 0 else gradient_accumulation_steps

    def update_params(self, loss: torch.Tensor):
        loss = self.scale_loss(loss)
        self.backward(loss)
        if self.should_update_params():
            self.step()
            self.zero_grad()

    def update_learning_rate(self):
        """ 更新learning rate """
        self.scheduler.step()

    def should_update_params(self) -> bool:
        """ 判断是否需要更新参数 """
        return (self._current_step % self.gradient_accumulation_steps == 0 or
                self._current_step == self._num_training_steps)

    def zero_grad(self):
        """ 梯度清零 """
        self.optimizer.zero_grad()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        loss_factor = self.gradient_accumulation_steps
        if (self._num_training_steps != -1 and
                self._current_step >= self._num_training_steps - self._remaining_steps):
            # 最后一次更新参数, 使用剩余步数作为缩放因子
            # 这里判断使用`>=`是因为更新参数先调用scale_loss方法后调用backward方法
            # 所以在这个方法中current_step时从0开始计数的, 最大为num_training_steps - 1
            loss_factor = self._remaining_steps

        loss = loss / loss_factor
        return loss

    def backward(self, loss: torch.Tensor):
        """ 反向传播求解梯度 """
        if self.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        self._current_step += 1

    def step(self):
        """ 梯度裁剪 & 更新权重参数 """
        self._clip_grad()
        if self.enable_amp:
            self.scaler.step(self.optimizer)  # 更新权重参数
            self.scaler.update()
        else:
            self.optimizer.step()  # 更新权重参数

    def _clip_grad(self):
        """ 梯度裁剪 """
        if self.gradient_clipping_max_norm is not None:
            if self.enable_amp:
                self.scaler.unscale_(self.optimizer)  # 将优化器中的梯度值反向缩放回原始值(取消缩放)

            """
            遍历所有参数组, 对每个参数组的参数进行裁剪, 不能写成:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'],
                self.gradient_clipping_max_norm,
            )
            如果优化器的params是以list传入(常见于AdamW优化参数模板), 这么写只会优化第一个元素中的参数
            """
            for param_group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    param_group['params'],
                    self.gradient_clipping_max_norm,
                )


if __name__ == '__main__':
    """
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    optim_scheduler = OptimSchedulerWrapper(
        optimizer,
        scheduler,
        gradient_clipping_max_norm=1.0,
        gradient_accumulation_steps=10,
        enable_amp=True,
        num_training_steps=98,
    )
    inputs = torch.randn(98, 1)
    targets = torch.randn(98, 1)
    for (data, label) in zip(inputs, targets):
        loss = torch.nn.MSELoss()(model(data), label)
        optim_scheduler.update_params(loss)
    """
