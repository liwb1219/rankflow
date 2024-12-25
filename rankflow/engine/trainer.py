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
from rankflow.utils.message_hub import MessageHub
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
        save_mode: Literal['epoch', 'step'] = 'epoch',
        save_interval: int = 1,
        logging_mode: Literal['epoch', 'step'] = 'epoch',
        logging_interval: int = 1,
        warmup_steps: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-5,
        scheduler_type: str = 'linear',
        gradient_clipping_max_norm: Optional[float] = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.message_hub = MessageHub()
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
        self._step = 0   # 当前训练步数
        self._save_mode = save_mode
        self._save_interval = save_interval
        self._logging_mode = logging_mode
        self._logging_interval = logging_interval

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
    def save_mode(self):
        return self._save_mode

    @property
    def save_interval(self):
        return self._save_interval

    @property
    def logging_mode(self):
        return self._logging_mode

    @property
    def logging_interval(self):
        return self._logging_interval

    def train(self):
        self.call_hooks('before_train')
        for epoch in range(self.max_epochs):
            self.train_epoch()
            self._epoch += 1
        self.call_hooks('after_train')

    def train_epoch(self):
        self.call_hooks('before_epoch')
        for data in self.train_dataloader:
            self.train_iter(data)
            self._step += 1
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

    def save_checkpoint(self, filename: str) -> None:




        os.makedirs(args.save_dir, exist_ok=True)
        model_name = '%d' % (epoch + 1) + '_model_' + format(results[args.preferential_metrics], '.3f') + '.bin'
        model_save_path = os.path.join(args.save_dir, model_name)
        torch.save(model.module.state_dict(), model_save_path)

        model_name = '%d' % (epoch + 1) + '_model_' + str(steps) + '.bin'
        model_save_path = os.path.join(args.save_dir, model_name)
        torch.save(model.module.state_dict(), model_save_path)


    def save_checkpoint2(self, file_name: str) -> None:
        """Save training state: ``epoch``, ``num_gpus``, ``model``, ``optimizer``, ``lr_scheduler``,
        ``metric_storage``, ``hooks`` (optional), ``grad_scaler`` (optional).

        Args:
            filename (str): The checkpoint will be saved as ``ckpt_dir/filename``.
        """
        data = {
            "num_gpus": get_world_size(),
            "model": self.model_or_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metric_storage": self.metric_storage,
        }
        data.update(dict(epoch=self.cur_epoch) if self.train_by_epoch else dict(iter=self.cur_iter))
        hook_states = {h.class_name: h.state_dict() for h in self._hooks if h.checkpointable}
        if hook_states:
            data["hooks"] = hook_states
        if self._enable_amp:
            data["grad_scaler"] = self._grad_scaler.state_dict()

        file_path = osp.join(self.ckpt_dir, file_name)
        logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

        # tag the latest checkpoint
        dst_file = osp.join(self.ckpt_dir, "latest.pth")
        symlink(file_name, dst_file)

    def save_checkpoint1(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: Optional[dict] = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Decide the number of epoch or iteration saved in
                checkpoint. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.setdefault('epoch', self.epoch + 1)
            meta.setdefault('iter', self.iter)
        else:
            meta.setdefault('epoch', self.epoch)
            meta.setdefault('iter', self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta':
            meta,
            'state_dict':
            weights_to_cpu(model.state_dict()),
            'message_hub':
            apply_to(self.message_hub.state_dict(),
                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = apply_to(
                    self.optim_wrapper.state_dict(),
                    lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            self.logger.warning(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers')
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(
            checkpoint,
            filepath,
            file_client_args=file_client_args,
            backend_args=backend_args)


if __name__ == '__main__':
    trainer = Trainer(model=nn.Linear(3, 3))
    pass
