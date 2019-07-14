import torch
from torch.utils.data import Dataset
import math
import itertools
import numpy as np

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

        num_batches = math.ceil(len(input_indices) / self.batch_size)

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
            if self.reverse:
                return [self.voc.word2index[w] for w in seq]
            else:
                return [self.voc.word2index[w] for w in seq] + [args.eos_index]

        def _msgstring2indices_(msg):
            return [int(c) for c in msg]
        
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


class ChooseDataset(Dataset):
    def __init__(
            self,  voc, 
            batch_size=args.batch_size, 
            dataset_file_path=args.train_file,
            device=args.device,
            d_num=args.num_distractors
        ):
        super().__init__()
        self.voc = voc
        self.batch_size = batch_size
        self.device = device
        self.file_path = dataset_file_path
        self.d_num = d_num # number of disctractors

        self.databatch_set = self.build_batches()
        self.batch_indices = np.arange(len(self.databatch_set))

    def __len__(self):
        return len(self.databatch_set)
    
    def __getitem__(self, idx):
        correct_batch = self.databatch_set[idx]

        distract_batches = []
        for _ in range(self.d_num):
            distract_batches.append(self.generate_distractor_batch(idx))
        
        return {
            'correct': correct_batch,
            'distracts': distract_batches
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return self.databatch_set[sample_idx]

    def reperm_batch(self, tgt_idx):
        batch_size = self.databatch_set[tgt_idx]['input'].shape[1]

        original_idx = torch.arange(batch_size, device=self.device)
        new_idx = torch.randperm(batch_size, device=self.device)

        while not (original_idx == new_idx).sum().eq(0):
            new_idx = torch.randperm(batch_size, device=self.device)

        shuffled_input = self.databatch_set[tgt_idx]['input'][:, new_idx]
        shuffled_mask = self.databatch_set[tgt_idx]['input_mask'][:, new_idx]
        shuffled_lens = self.databatch_set[tgt_idx]['input_lens'][new_idx]
        max_len = self.databatch_set[tgt_idx]['input_max_len']

        return {
            'input': shuffled_input,
            'input_mask': shuffled_mask,
            'input_lens': shuffled_lens,
            'input_max_len': max_len
        }

    def string_set2input_target_indices(self, string_set):
        input_indices = []
        def _string2indices_(seq):
            return [self.voc.word2index[w] for w in seq]
        for string in string_set:
            input_indices.append(_string2indices_(string))
        
        return input_indices
        

    def build_tensor_mask_lens_maxlen(self, indices_batch, value=args.pad_index):
        padded_indices = FruitSeqDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(self.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = FruitSeqDataset.build_mask(padded_indices, value)
        mask = torch.ByteTensor(mask).to(self.device)

        padded_indices = torch.LongTensor(padded_indices).to(self.device)
        
        return padded_indices, mask, lens, max_len

    def build_batches(self):
        batches = []

        string_set = FruitSeqDataset.load_stringset(self.file_path)
        input_indices = self.string_set2input_target_indices(string_set)

        num_batches = math.ceil(len(input_indices) / self.batch_size)

        for i in range(num_batches):
            input_indices_batch = input_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(input_indices))]
            input_indices_batch.sort(key=len, reverse=True)

            in_var, in_mask, in_len, in_max_len = \
                self.build_tensor_mask_lens_maxlen(input_indices_batch)

            batches.append({
                'input': in_var,
                'input_mask': in_mask,
                'input_lens': in_len,
                'input_max_len': in_max_len
            })

        return batches


class ChoosePairDataset(Dataset):
    def __init__(
        self,  voc, 
        batch_size=args.batch_size, 
        dataset_file_path=args.train_file,
        device=args.device,
        d_num=args.num_distractors
    ):
        super().__init__()
        self.voc = voc
        self.batch_size = batch_size
        self.device = device
        self.file_path = dataset_file_path
        self.d_num = d_num # number of disctractors

        self.databatch_set = self.build_batches()
        self.batch_indices = np.arange(len(self.databatch_set))

    def __len__(self):
        return len(self.databatch_set)

    def __getitem__(self, idx):
        correct_batch = self.databatch_set[idx]

        distract_batches = []
        for _ in range(self.d_num):
            distract_batches.append(self.generate_distractor_batch(idx))
        
        return {
            'correct': correct_batch,
            'distracts': distract_batches
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return self.databatch_set[sample_idx]

    def reperm_batch(self, tgt_idx):
        batch_size = self.databatch_set[tgt_idx]['message'].shape[1]

        original_idx = torch.arange(batch_size, device=self.device)
        new_idx = torch.randperm(batch_size, device=self.device)

        while not (original_idx == new_idx).sum().eq(0):
            new_idx = torch.randperm(batch_size, device=self.device)

        shuffled_seq = self.databatch_set[tgt_idx]['sequence'][:, new_idx]
        shuffled_seq_mask = self.databatch_set[tgt_idx]['seq_mask'][:, new_idx]
        max_len = self.databatch_set[tgt_idx]['seq_max_len']

        return {
            'sequence': shuffled_seq,
            'seq_mask': shuffled_seq_mask,
            'seq_max_len': max_len
        }

    def pair_set2msg_io_indices(self, pair_set:list) -> list:
        msg_indices = []
        io_indices = []

        def _iostring2indices_(seq):
            return [self.voc.word2index[w] for w in seq]

        def _msgstring2indices_(msg):
            return [int(c) for c in msg]
        
        for pair in pair_set:
            msg_indices.append(_msgstring2indices_(pair[0]))
            io_indices.append(_iostring2indices_(pair[1]))
        
        return msg_indices, io_indices

    def build_tensor_mask_lens_maxlen(self, indices_batch, value=args.pad_index):
        padded_indices = PairDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(self.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = PairDataset.build_mask(padded_indices, value)
        mask = torch.ByteTensor(mask).to(self.device)

        padded_indices = torch.LongTensor(padded_indices).to(self.device)
        
        return padded_indices, mask, lens, max_len

    def build_batches(self) -> list:
        batches = []

        pair_set = PairDataset.load_pairset(self.file_path)
        msg_indices, io_indices = self.pair_set2msg_io_indices(pair_set)

        num_batches = math.ceil(len(msg_indices) / self.batch_size)

        for i in range(num_batches):
            msg_indices_batch = msg_indices[i*self.batch_size:
                                    min((i+1)*self.batch_size, len(io_indices))]
            seq_indices_batch = io_indices[i*self.batch_size:
                                        min((i+1)*self.batch_size, len(msg_indices))]

            msg_var, msg_mask, msg_len, _ = \
                self.build_tensor_mask_lens_maxlen(msg_indices_batch, value=-1)
            seq_var, seq_mask, _, seq_max_len = \
                self.build_tensor_mask_lens_maxlen(seq_indices_batch)

            batches.append({
                'message': msg_var,
                'msg_mask': msg_mask,
                'msg_lens': msg_len,
                'sequence': seq_var,
                'seq_mask': seq_mask,
                'seq_max_len': seq_max_len
            })

        return batches


if __name__ == '__main__':
    voc = Voc()
    batchset = ChoosePairDataset(voc, dataset_file_path='./data/2_perfect/all_data.txt')

    print('correct batch:')
    print(batchset[0]['correct'])

    print('first distract:')
    print(batchset[0]['distracts'][0])
