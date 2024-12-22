# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from typing import Any, Optional


class MessageHub:
    """
    消息中心类, 用于存储和读取信息

    功能:
    - 存储或更新信息
    - 获取信息, 支持指定默认值
    """
    def __init__(self):
        self._storage = {}

    def update_info(self, key: str, value: Any) -> None:
        self._storage[key] = value

    def get_info(self, key: str, default: Optional[Any] = None) -> Any:
        return self._storage.get(key, default)
