import torch
from torch.utils.data import Dataset
import math
import itertools

from utils.conf import args
from preprocesses.Voc import Voc

class FruitSeqDataset(Dataset):
    def __init__(
        self, voc, 
        batch_size=args.batch_size, 
        dataset_file_path=args.train_file,
        device=args.device
    ):
        self.voc = voc
        self.batch_size = batch_size
        self.device = device

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

    def build_tensor_mask_lens_maxlen(self, indices_batch):
        padded_indices = FruitSeqDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(self.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = FruitSeqDataset.build_mask(padded_indices)
        mask = torch.ByteTensor(mask).to(self.device)

        padded_indices = torch.LongTensor(padded_indices).to(self.device)
        
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
                self.build_tensor_mask_lens_maxlen(input_indices_batch)
            tgt_var, tgt_mask, _, tgt_max_len = \
                self.build_tensor_mask_lens_maxlen(target_indices_batch)

            batches.append({
                'input': in_var,
                'input_mask': in_mask,
                'input_lens': in_len,
                'target': tgt_var,
                'target_mask': tgt_mask,
                'target_max_len': tgt_max_len
            })

        return batches


class PairDataset(Dataset):
    def __init__(
            self, voc, reverse=False, 
            batch_size=args.batch_size,
            dataset_file_path=args.train_file,
            device=args.device
        ):
        self.voc = voc
        self.batch_size = batch_size
        # T for (msg, seq) as io, F for (seq, msg) as io
        self.reverse = reverse
        self.device = device

        self.batches = self.build_batches(dataset_file_path)

    def __len__(self):
        return len(self.batches)
        
    def __getitem__(self, idx):
        return self.batches[idx]

    @staticmethod
    def load_pairset(file_path:str) -> list:
        f = open(file_path, 'r')

        pair_set = []
        for line in f.readlines():
            line = line.strip()
            msg, ioseq = line.split('\t')
            pair_set.append([msg, ioseq])

        return pair_set

    def pair_set2msg_io_indices(self, pair_set:list) -> list:
        msg_indices = []
        io_indices = []

        def _iostring2indices_(seq):
            return [self.voc.word2index[w] for w in seq] + [args.eos_index]

        def _msgstring2indices_(msg):
            return [int(c) for c in msg] + [args.msg_vocsize-1]
        
        for pair in pair_set:
            # input contains neither SOS or EOS, target contains EOS
            msg_indices.append(_msgstring2indices_(pair[0]))
            io_indices.append(_iostring2indices_(pair[1]))
        
        return msg_indices, io_indices

    @staticmethod
    def pad(l, fillvalue=args.pad_index):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    @staticmethod
    def build_mask(l, value=args.pad_index):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for index in seq:
                if index == value:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def build_tensor_mask_lens_maxlen(self, indices_batch, value=args.pad_index):
        padded_indices = PairDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(self.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = PairDataset.build_mask(padded_indices, value)
        mask = torch.ByteTensor(mask).to(self.device)

        padded_indices = torch.LongTensor(padded_indices).to(self.device)
        
        return padded_indices, mask, lens, max_len

    def build_batches(self, file_path:str) -> list:
        batches = []

        pair_set = self.load_pairset(file_path)
        msg_indices, io_indices = self.pair_set2msg_io_indices(pair_set)

        num_batches = math.ceil(len(msg_indices) / self.batch_size)

        for i in range(num_batches):
            if self.reverse:
                input_indices_batch = io_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(io_indices))]
                target_indices_batch = msg_indices[i*self.batch_size:
                                            min((i+1)*self.batch_size, len(msg_indices))]
            else:
                input_indices_batch = msg_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(io_indices))]
                target_indices_batch = io_indices[i*self.batch_size:
                                            min((i+1)*self.batch_size, len(msg_indices))]

            input_indices_batch.sort(key=len, reverse=True)
            target_indices_batch.sort(key=len, reverse=True)

            if self.reverse:
                in_var, in_mask, in_len, _ = \
                    self.build_tensor_mask_lens_maxlen(input_indices_batch)
                tgt_var, tgt_mask, _, tgt_max_len = \
                    self.build_tensor_mask_lens_maxlen(target_indices_batch, value=-1)
            else:
                in_var, in_mask, in_len, _ = \
                    self.build_tensor_mask_lens_maxlen(input_indices_batch, value=-1)
                tgt_var, tgt_mask, _, tgt_max_len = \
                    self.build_tensor_mask_lens_maxlen(target_indices_batch)

            batches.append({
                'input': in_var,
                'input_mask': in_mask,
                'input_lens': in_len,
                'target': tgt_var,
                'target_mask': tgt_mask,
                'target_max_len': tgt_max_len
            })

        return batches


if __name__ == '__main__':
    voc = Voc()
    batchset = PairDataset(voc, reverse=False)

    print('input:')
    print(batchset[5]['input'])
    print('input shape:')
    print(batchset[5]['input'].shape)
    print('input_mask:')
    print(batchset[5]['input_mask'])
    print('input_mask shape:')
    print(batchset[5]['input_mask'].shape)
    print('input_lens')
    print(batchset[5]['input_lens'])

    print('target:')
    print(batchset[5]['target'])
    print('target mask:')
    print(batchset[5]['target_mask'])
    print('target_max_len:')
    print(batchset[5]['target_max_len'])
