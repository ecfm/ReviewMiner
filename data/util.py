import os

import pandas as pd
import torch
import torch.utils.data as data

from data.amazon_dataset import AmazonDataset

uid_dict = {}
uid_list = []
pid_dict = {}
pid_list = []

class Preprocessor_base():
    def __init__(self):
        self.fn = None

    def make_fn(self):
        raise NotImplementedError()

    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e


class Preprocessor(Preprocessor_base):
    def __init__(self, tokenizer, seq_len, data_type):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_type = data_type

    def make_fn(self):
        return lambda input: self.tokenizer.encode(input)

def read_csv(filename, chunksize=1000000):
  reader = pd.read_csv(filename, chunksize=chunksize)
  df = pd.DataFrame()
  for chunk in reader:
    df = pd.concat([df, chunk])
  return df



def collate_fn(samples):
    """ Creates a batch out of samples """
    uid_tensor = torch.LongTensor([s['u_id'] for s in samples])
    pid_tensor = torch.LongTensor([s['p_id'] for s in samples])
    max_len = max(map(lambda s: len(s['review']), samples))
    # Zero pad mask
    input_mask = torch.ByteTensor([[1] * len(s['review']) + [0] * (max_len - len(s['review'])) for s in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    input = torch.LongTensor([s['review'] + [50256] * (max_len - len(s['review'])) for s in samples])

    return uid_tensor, pid_tensor, input[:, :-1], input[:, 1:].contiguous(), input_mask[:, 1:]

def prepare_dataset(data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, data_type='t0', num_workers=1, make_train=True, make_val=True, make_test=True):
    # data_dir, dataset_name, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, num_workers = args.data_dir, args.dataset, tokenizer, batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1], batch_schedule[-1][0], batch_schedule[-1][1], args.workers

    loaders = []
    if dataset_name == 'am':
        train_collate_fn = collate_fn
        val_collate_fn = collate_fn
        test_collate_fn = collate_fn

        if make_train:
            train_preproc = Preprocessor(tokenizer, train_seq_len, data_type)
            d_train = AmazonDataset(os.path.join(data_dir, 'sample.csv'), train_preproc)
            print('Train dataset size', len(d_train))
            loaders.append(data.DataLoader(d_train,
                                           # sampler=DistributedSampler(d_train) if distributed else None,
                                           batch_size=train_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=train_collate_fn) if d_train else None)
        if make_val:
            val_preproc = Preprocessor(tokenizer, val_seq_len, data_type)
            d_val = AmazonDataset(os.path.join(data_dir, 'sample.csv'), val_preproc)
            print('Val dataset size', len(d_val))
            loaders.append(data.DataLoader(d_val,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=val_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn) if d_val else None)
        if make_test:
            test_preproc = Preprocessor(tokenizer, test_seq_len, data_type)
            d_test = AmazonDataset(os.path.join(data_dir, 'sample.csv'), test_preproc)
            print('Test dataset size', len(d_test))
            loaders.append(data.DataLoader(d_test,
                                           # sampler=DistributedSampler(d_val),
                                           batch_size=test_bsz,
                                           pin_memory=True,
                                           drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=test_collate_fn) if d_test else None)
    else:
        raise Exception('Invalid dataset')

    return loaders
