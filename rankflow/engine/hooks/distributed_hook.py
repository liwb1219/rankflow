# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from .hookbase import HookBase
from .priority import HookPriority
from typing import Tuple
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP


class DistributedHook(HookBase):
    priority = HookPriority.HIGH

    def before_train(self):
        local_rank, rank, world_size = self.init_distributed_environment()
        # 使用 property setter 更新 Trainer 的相关属性
        self.trainer.local_rank = local_rank
        self.trainer.rank = rank
        self.trainer.world_size = world_size

        # 将模型包装为DistributedDataParallel(DDP)模型以实现分布式训练
        self.trainer.model = DDP(
            self.trainer.model.to(local_rank),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=self.trainer.find_unused_parameters,
        )

    def before_epoch(self):
        if hasattr(self.trainer.train_dataloader.sampler, 'set_epoch'):
            self.trainer.train_dataloader.sampler.set_epoch(self.trainer.epoch)

    @staticmethod
    def init_distributed_environment(backend: str = 'nccl', init_method: str = 'env://') -> Tuple[int, int, int]:
        """
        :param backend: 分布式后端, GPU的分布式训练用NCCL
        :param init_method: 初始化方法, 默认为'env://', 表示使用环境变量进行初始化, 可以从环境变量中读取分布式的信息(os.environ)
        :return: local_rank(当前进程的本地rank), rank(当前进程的全局rank), world_size(总的进程数)
        """
        try:
            dist.init_process_group(backend=backend, init_method=init_method)
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f'\033[1;32mInitialized distributed environment with '
                  f'\033[1;36mLOCAL_RANK {local_rank}, RANK {rank}, WORLD_SIZE {world_size}\033[0m')
            return local_rank, rank, world_size
        except Exception as e:
            raise RuntimeError(
                f'\033[1;33mPlease use \033[1;32mtorchrun \033[1;33mto launch the script. '
                f'{e}\033[0m'
            )
