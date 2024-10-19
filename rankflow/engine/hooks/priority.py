# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from enum import Enum
from typing import Union


class HookPriority(Enum):
    """
    +---------+-------+
    |  Level  | Value |
    +---------+-------+
    | LOWEST  |  10   |
    |   LOW   |  20   |
    | NORMAL  |  30   |
    |  HIGH   |  40   |
    | HIGHEST |  50   |
    +---------+-------+
    """
    LOWEST = 10
    LOW = 20
    NORMAL = 30
    HIGH = 40
    HIGHEST = 50


def get_priority(priority: Union[int, str, HookPriority]) -> int:
    if isinstance(priority, int):
        if 10 <= priority <= 50:  # 检查整数范围
            return priority
        else:
            raise ValueError(
                f"Integer priority must be between 10 and 50."
            )
    elif isinstance(priority, str):
        try:
            return HookPriority[priority.upper()].value
        except KeyError:
            raise ValueError(
                f"Invalid priority string: '{priority}'. Must be one of {list(HookPriority.__members__.keys())}."
            )
    elif isinstance(priority, HookPriority):
        return priority.value
    else:
        raise TypeError(
            f"Expected type 'int | str | HookPriority', got '{type(priority).__name__}' instead."
        )


if __name__ == '__main__':
    from tabulate import tabulate
    headers = ['Level', 'Value']
    tabular_data = []
    for p in HookPriority:
        tabular_data.append((p.name, p.value))

    table = tabulate(tabular_data, headers, tablefmt='pretty')
    print(table)

    print(get_priority('NORMAL'))
