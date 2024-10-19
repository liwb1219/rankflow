# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

class DataPreprocessor:
    def __init__(self):
        pass

    def __call__(self, x):
        print(f'hello {x}')


if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    data_preprocessor('liwenbiao')
