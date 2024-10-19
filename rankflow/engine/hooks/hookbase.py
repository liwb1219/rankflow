# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

import weakref
from typing import Optional, Union
from .priority import HookPriority


class HookBase:
    """
    HookBase 类定义了在训练、验证和推理过程中各个阶段的回调方法
    子类可以继承此类并重写相应的方法来实现在特定阶段执行的操作: 如日志记录、模型保存、评估等
    """
    trainer: Optional[weakref.ProxyType] = None  # 指向Trainer的弱引用, 在注册钩子时由Trainer设置
    priority: Union[int, str, HookPriority] = HookPriority.NORMAL

    def before_train(self):
        """
        在训练开始之前调用的方法
        这个方法可以在训练循环启动之前做一些初始化工作
        """
        pass

    def before_epoch(self):
        """
        在每个epoch开始之前调用的方法
        这个方法可以在每个epoch开始之前执行一些操作, 如重置某些状态
        """
        pass

    def before_iter(self):
        """
        在每次迭代开始之前调用的方法
        这个方法可以在处理每个数据批次之前执行一些操作, 如预处理输入数据
        """
        pass

    def after_iter(self):
        """
        在每次迭代结束之后调用的方法
        这个方法可以在处理完一个数据批次之后执行一些操作, 如记录日志或更新状态
        """
        pass

    def after_epoch(self):
        """
        在每个epoch结束之后调用的方法
        这个方法可以在每个epoch结束之后执行一些操作, 如保存模型或评估效果
        """
        pass

    def after_train(self):
        """
        在训练结束之后调用的方法
        这个方法可以在训练循环结束之后执行一些清理工作或最终操作
        """
        pass
