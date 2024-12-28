# -*- coding: utf-8 -*-
# Copyright (c) 2024 liwenbiao. All rights reserved.

from torch.utils.data import Dataset, IterableDataset


class MapDataReader(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class IterableDataReader(IterableDataset):
    def __init__(self):
        pass

    def __iter__(self):
        pass


class MapCrossEncoderDatasetReader(Dataset):
    def __init__(self, data_path, max_len, mode='train', rank=None):
        self.max_len = max_len
        self.tokenizer = FullTokenizer()

        self.data = []
        feather_data_dir = Path(data_path).resolve().parent.joinpath('feather')
        feather_data_path = feather_data_dir.joinpath(Path(data_path).resolve().name.replace('txt', 'feather'))
        if feather_data_path.exists():
            self.data = pd.read_feather(feather_data_path)
            self.data = self.data.values.tolist()
            print('\033[1;35m-------------- Read data from feats --------------')
            print(f'{mode} data size: {len(self.data)}')
            print('--------------------------------------------------\033[0m')
        else:
            data_list = []
            data_size = 0
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_size += 1
                    original_query, query, trend, label, source = preprocess_input_data(
                        data=line, mode=mode, multi_span=True,
                    )
                    data_list.append({
                        'original_query': original_query,
                        'query': query,
                        'trend': trend,
                        'label': label,
                        'source': source,
                    })
                    self.data.append((original_query, query, trend, label, source))
            print('\033[1;35m-------------- Read data from files --------------')
            print(f'{mode} data size: {data_size}')
            print('--------------------------------------------------\033[0m')
            if rank == 0:
                feather_data_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(data_list).to_feather(feather_data_path)

    def __getitem__(self, item):
        original_query, query, trend, label, source = self.data[item]
        inputs = self.tokenizer.encode_plus(
            text=query,
            text_pair=trend,
            add_special_tokens=True,
            padding_to_max_length=True,
            max_length=self.max_len,
            do_word_input=True,
        )
        return {
            'query': original_query,
            'input_ids': torch.tensor(inputs['input_ids']),
            'token_type_ids': torch.tensor(inputs['token_type_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'labels': torch.tensor(label),
            'source': torch.tensor(source),
        }

    def __len__(self):
        return len(self.data)


class Stage1IterableDatasetReader(IterableDataset):
    def __init__(self, args, rank, world_size):
        self.train_data_path = args.train_data_path
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = FullTokenizer()
        print('\033[1;32m--------------------- DEBUG ----------------------')
        print('debug word tokenize:', self.tokenizer.word_tokenize('15岁平衡水油护肤'))
        print('debug do word input encode:', self.tokenizer.encode('15岁平衡水油护肤', do_word_input=True))
        print('--------------------------------------------------\033[0m')
        self.args = args

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            worker_num = 1
        else:
            worker_id = worker_info.id
            worker_num = worker_info.num_workers
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            for idx, data in enumerate(f):
                if idx % (self.world_size * worker_num) == self.rank * worker_num + worker_id:
                    yield self.parse_data(data)

    def parse_data(self, data):
        trend = preprocess_input_data(data=data)
        inputs = self.tokenizer.encode_plus(trend, do_word_input=True)
        mlm_inputs = construct_mlm_inputs(
            inputs=inputs,
            mlm_prob=self.args.mlm_prob,
            pad_token_id=0,
            sep_token_id=102,
            mask_token_id=103,
            max_len=self.args.max_len,
        )

        outputs = {
            'input_ids': torch.tensor(mlm_inputs['input_ids']),
            'attention_mask': torch.tensor(mlm_inputs['attention_mask']),
            'token_type_ids': torch.tensor(mlm_inputs['token_type_ids']),
            'labels': torch.tensor(mlm_inputs['labels']),
        }
        return outputs

