# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from typing import Union, Optional, List, Dict, Tuple, Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import get_scheduler

import weakref

from rankflow.engine.hooks import HookBase
from rankflow.engine.hooks import HookPriority, get_priority
from rankflow.utils.logger import setup_logger
from rankflow.optim import OptimSchedulerWrapper


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        work_dir: str = 'outputs',
        enable_amp: bool = True,

        batch_size: int = 64,

        train_dataloader: Optional[Union[DataLoader, List]] = None,
        valid_dataloader: Optional[Union[DataLoader, List]] = None,

        log_level: Union[int, str] = 'DEBUG',
        log_file: Optional[str] = None,
        log_file_mode: Optional[str] = None,
        enable_highlight_colors: bool = False,

        enable_ddp: bool = True,
        find_unused_parameters: bool = False,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        save_steps: Optional[int] = 1,
        logging_steps: Optional[int] = 1,
        warmup_steps: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-5,
        scheduler_type: str = 'linear',
        gradient_clipping_max_norm: Optional[float] = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self._hooks: List[HookBase] = []
        self._local_rank = None
        self._rank = None
        self._world_size = None

        self.model = model
        self._work_dir = work_dir
        self._enable_amp = enable_amp

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.logger = setup_logger(
            log_level=log_level,
            log_file=log_file,
            file_mode=log_file_mode,
            enable_highlight_colors=enable_highlight_colors,
        )

        self._enable_ddp = enable_ddp
        self._find_unused_parameters = find_unused_parameters
        self._max_iters = max_iters
        self._max_epochs = max_epochs
        self._epoch = 0  # 当前训练轮次
        self._step = 0  # 当前训练步数
        self._save_steps = save_steps
        self._logging_steps = logging_steps

        optimizer, scheduler, num_training_steps = self.build_optimizer_and_scheduler(
            model=model,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon,
            scheduler_type=scheduler_type,
        )

        self.optim_scheduler = OptimSchedulerWrapper(
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clipping_max_norm=gradient_clipping_max_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_amp=enable_amp,
            num_training_steps=num_training_steps,
        )

    @property
    def local_rank(self):
        return self._local_rank

    @local_rank.setter
    def local_rank(self, value):
        self._local_rank = value

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        self._rank = value

    @property
    def world_size(self):
        return self._world_size

    @world_size.setter
    def world_size(self, value):
        self._world_size = value

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def enable_amp(self):
        return self._enable_amp

    @property
    def enable_ddp(self):
        return self._enable_ddp

    @property
    def find_unused_parameters(self):
        return self._find_unused_parameters

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def epoch(self):
        return self._epoch

    @property
    def step(self):
        return self._step

    @property
    def save_steps(self):
        return self._save_steps

    @property
    def logging_steps(self):
        return self._logging_steps

    def train(self):
        self.call_hooks('before_train')
        for epoch in range(self.max_epochs):
            self.train_epoch()
        self.call_hooks('after_train')

    def train_epoch(self):
        self.call_hooks('before_epoch')
        for data in self.train_dataloader:
            self.train_iter(data)
        self.call_hooks('after_epoch')

    def train_iter(self, data):
        self.call_hooks('before_iter')
        self.model.train()
        batch = {k: v for k, v in data.items()}

        if self.enable_amp:
            with autocast():
                loss = self.model(**batch)['loss']
        else:
            loss = self.model(**batch)['loss']

        self.optim_scheduler.update_params(loss)
        self.optim_scheduler.update_lr()
        self.call_hooks('after_iter')

    def call_hooks(self, fn_name: str) -> None:
        for hook in self._hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)()

    def register_hook(
        self,
        hook: HookBase,
        priority: Optional[Union[int, str, HookPriority]] = None,
    ) -> None:
        assert isinstance(hook, HookBase), 'hook must be an instance of HookBase'
        if priority is not None:
            hook.priority = priority

        # 使用weakref.proxy创建当前Trainer的弱引用, 并将其赋值给Hook的trainer属性
        # 使用弱引用的好处是可以避免循环引用导致的内存泄漏问题
        # 这样Hook可以安全地访问Trainer, 但不会阻止Trainer被垃圾回收
        hook.trainer = weakref.proxy(self)

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook.priority) <= get_priority(self._hooks[i].priority):
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_optimizer_and_scheduler(
        self,
        model: nn.Module,
        warmup_steps: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-5,
        scheduler_type: str = 'linear',
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int]:
        if isinstance(warmup_steps, int) and warmup_ratio is None:
            num_training_steps = self.max_iters
            num_warmup_steps = warmup_steps
        elif isinstance(warmup_ratio, float) and warmup_steps is None:
            num_training_steps = self.max_epochs * len(self.train_dataloader)
            num_warmup_steps = warmup_ratio * num_training_steps
        else:
            param_1 = f'\033[1;33m"warmup_steps"\033[0m'  # 黄色加粗
            param_2 = f'\033[1;33m"warmup_ratio"\033[0m'  # 黄色加粗
            raise ValueError(
                f'\033[1;31mExactly one of {param_1} \033[1;31mor {param_2} \033[1;31mmust be specified.\033[0m'
            )

        optimizer, scheduler = self._build_optimizer_and_scheduler(
            model=model,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon,
            scheduler_type=scheduler_type,
        )
        return optimizer, scheduler, num_training_steps

    @staticmethod
    def _build_optimizer_and_scheduler(
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


if __name__ == '__main__':
    trainer = Trainer(model=nn.Linear(3, 3))
    pass
