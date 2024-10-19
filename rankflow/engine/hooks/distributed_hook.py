# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from .hookbase import HookBase
from .priority import HookPriority


class DistributedHook(HookBase):
    priority = HookPriority.HIGH
