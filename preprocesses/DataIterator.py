import torch
from torch.utils.data import Dataset
import math
import itertools

from utils.conf import args

class  FruitSeqDataset(Dataset):
    def __init__(self, voc, batch_size=args.batch_size, dataset_file_path=args.train_file):
        self.voc = voc
        self.batch_size = batch_size
        self.batches = self.build_batches(dataset_file_path)

    def __len__(self):
        return len(self.batches)
        
    def __getitem__(self, idx):
        return self.batches[idx]

    @staticmethod
    def load_stringset(file_path):
        f = open(file_path, 'r')

        string_set = []
        for line in f.readlines():
            string_set.append(line.strip())

        return string_set

    def string_set2input_target_indices(self, string_set):
        input_indices = []
        target_indices = []

        def _string2indices_(seq):
            return [self.voc.word2index[w] for w in seq] + [args.eos_index]
        
        for string in string_set:
            # input contains neither SOS or EOS, target contains EOS
            input_indices.append(_string2indices_(string)[:-1])
            target_indices.append(_string2indices_(string))
        
        return input_indices, target_indices

    @staticmethod
    def pad(l, fillvalue=args.pad_index):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    @staticmethod
    def build_mask(l, value=args.pad_index):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for index in seq:
                if index == args.pad_index:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    @staticmethod
    def build_tensor_mask_lens_maxlen(indices_batch):
        padded_indices = FruitSeqDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(args.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = FruitSeqDataset.build_mask(padded_indices)
        mask = torch.ByteTensor(mask).to(args.device)

        padded_indices = torch.LongTensor(padded_indices).to(args.device)
        
        return padded_indices, mask, lens, max_len


    def build_batches(self, file_path):
        batches = []

        string_set = self.load_stringset(file_path)
        input_indices, target_indices = self.string_set2input_target_indices(string_set)

        num_batches = num_batches = math.ceil(len(input_indices) / self.batch_size)

        for i in range(num_batches):
            input_indices_batch = input_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(input_indices))]
            target_indices_batch = target_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(input_indices))]
            input_indices_batch.sort(key=len, reverse=True)
            target_indices_batch.sort(key=len, reverse=True)

            in_var, in_mask, in_len, _ = \
                FruitSeqDataset.build_tensor_mask_lens_maxlen(input_indices_batch)
            tgt_var, tgt_mask, _, tgt_max_len = \
                FruitSeqDataset.build_tensor_mask_lens_maxlen(target_indices_batch)

            batches.append({
                'input': in_var,
                'input_mask': in_mask,
                'input_lens': in_len,
                'target': tgt_var,
                'target_mask': tgt_mask,
                'target_max_len': tgt_max_len
            })

        return batches
