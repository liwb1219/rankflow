# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from .hookbase import HookBase
from .priority import HookPriority


class CheckpointHook(HookBase):
    priority = HookPriority.LOWEST

    def after_iter(self):
        if self.trainer.save_mode == 'step' and self.trainer.step % self.trainer.save_interval == 0:
            self.trainer.save_checkpoint()

    def after_epoch(self):
        if self.trainer.save_mode == 'epoch' and self.trainer.epoch % self.trainer.save_interval == 0:
            self.trainer.save_checkpoint(model_dir=f'checkpoint-{self.trainer.epoch}')
