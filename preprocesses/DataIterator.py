import torch
from torch.utils.data import Dataset
import math
import itertools
import numpy as np
import random
import os
import torchvision
import copy
from PIL import Image

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

        candidate_batches = []
        golden_idx = random.randint(0, self.d_num)
        for i in range(self.d_num+1):
            if i == golden_idx:
                candidate_batches.append(self.databatch_set[idx])
            else:
                candidate_batches.append(self.generate_distractor_batch(idx))
        
        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        while self.batch_size == 1 and sample_idx == tgt_idx:
            sample_idx = np.random.choice(self.batch_indices)
        
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return copy.deepcopy(self.databatch_set[sample_idx])

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

        # ceil/floor
        if len(input_indices) < self.batch_size:
            num_batches = 1
        else:
            num_batches = math.floor(len(input_indices) / self.batch_size)

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

        candidate_batches = []
        golden_idx = random.randint(0, self.d_num)
        for i in range(self.d_num+1):
            if i == golden_idx:
                candidate_batches.append(self.databatch_set[idx])
            else:
                candidate_batches.append(self.generate_distractor_batch(idx))
        
        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return copy.deepcopy(self.databatch_set[sample_idx])

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

        if len(io_indices) < self.batch_size:
            num_batches = 1
        else:
            num_batches = math.floor(len(io_indices) / self.batch_size)

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


class ImgChooseDataset(Dataset):
    def __init__(
            self,
            batch_size=args.batch_size, 
            dataset_dir_path=args.train_file,
            device=args.device,
            d_num=args.num_distractors
        ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.dir_path = dataset_dir_path
        self.d_num = d_num # number of disctractors

        self.databatch_set = self.build_batches()
        self.batch_indices = np.arange(len(self.databatch_set))

    def __len__(self):
        return len(self.databatch_set)
    
    def __getitem__(self, idx):
        correct_batch = self.databatch_set[idx]

        candidate_batches = [
            self.generate_distractor_batch(idx) for _ in range(self.d_num+1)
        ]

        golden_idx = np.random.randint(0, high=self.d_num+1, size=(correct_batch['imgs'].shape[0]))

        for i in range(correct_batch['imgs'].shape[0]):
            candidate_batches[golden_idx[i]]['imgs'][i, :, :, :] = correct_batch['imgs'][i, :, :, :]
            candidate_batches[golden_idx[i]]['label'][i] = correct_batch['label'][i]

        golden_idx = torch.from_numpy(golden_idx).to(self.device).to(torch.long)

        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        while self.batch_size == 1 and sample_idx == tgt_idx:
            sample_idx = np.random.choice(self.batch_indices)
        
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return copy.deepcopy(self.databatch_set[sample_idx])

    def reperm_batch(self, tgt_idx):
        batch_size = self.databatch_set[tgt_idx]['imgs'].shape[0]

        original_idx = torch.arange(batch_size, device=self.device)
        new_idx = torch.randperm(batch_size, device=self.device)

        while not (original_idx == new_idx).sum().eq(0):
            new_idx = torch.randperm(batch_size, device=self.device)

        shuffled_imgs = self.databatch_set[tgt_idx]['imgs'][new_idx]
        shuffled_labels = [self.databatch_set[tgt_idx]['label'][i] for i in new_idx]

        return {
            'imgs': shuffled_imgs,
            'label': shuffled_labels,
        }

    @staticmethod
    def load_img_set(dir_path):
        img_file_names = os.listdir(dir_path)
        imgs = [Image.open(os.path.join(dir_path, name)).convert('RGB') for name in img_file_names]
        return img_file_names, imgs

    @staticmethod
    def build_img_tensors(imgs, device=args.device):
        tensors = []
        for img in imgs:
            tensors.append(torchvision.transforms.ToTensor()(img))
        tensors = torch.stack(tensors).to(device)
        return tensors

    def build_batches(self):
        batches = []

        img_names, imgs = ImgChooseDataset.load_img_set(self.dir_path)
        img_names = [name.split('.')[0] for name in img_names]

        assert len(img_names) == len(imgs)
        c = list(zip(img_names, imgs))
        img_names, imgs = zip(*c)
        img_names = list(img_names)
        imgs = list(imgs)

        # ceil/floor
        if len(imgs) < self.batch_size:
            num_batches = 1
        else:
            num_batches = math.floor(len(imgs) / self.batch_size)

        for i in range(num_batches):
            img_batch = ImgChooseDataset.build_img_tensors(
                    imgs[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))],
                    device=self.device
                )
            img_label_batch = img_names[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))]

            batches.append({
                'imgs': img_batch,
                'label': img_label_batch,
            })

        return batches


class ImgChoosePairDataset(Dataset):
    def __init__(
            self,
            batch_size=args.batch_size, 
            dataset_dir_path=args.train_file,
            lan_file_path=args.data_file,
            device=args.device,
            d_num=args.num_distractors
        ):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.dir_path = dataset_dir_path
        self.lan_path = lan_file_path
        self.d_num = d_num # number of disctractors

        self.databatch_set = self.build_batches()
        self.batch_indices = np.arange(len(self.databatch_set))

    def __len__(self):
        return len(self.databatch_set)
    
    def __getitem__(self, idx):
        correct_batch = self.databatch_set[idx]

        candidate_batches = [
            self.generate_distractor_batch(idx) for _ in range(self.d_num+1)
        ]

        golden_idx = np.random.randint(0, high=self.d_num+1, size=(correct_batch['imgs'].shape[0]))

        for i in range(correct_batch['imgs'].shape[0]):
            candidate_batches[golden_idx[i]]['imgs'][i, :, :, :] = correct_batch['imgs'][i, :, :, :]
            candidate_batches[golden_idx[i]]['label'][i] = correct_batch['label'][i]

        golden_idx = torch.from_numpy(golden_idx).to(self.device).to(torch.long)

        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def generate_distractor_batch(self, tgt_idx):
        sample_idx = np.random.choice(self.batch_indices)
        while self.batch_size == 1 and sample_idx == tgt_idx:
            sample_idx = np.random.choice(self.batch_indices)
        
        if sample_idx == tgt_idx:
            return self.reperm_batch(tgt_idx)
        else:
            return copy.deepcopy(self.databatch_set[sample_idx])

    def reperm_batch(self, tgt_idx):
        batch_size = self.databatch_set[tgt_idx]['imgs'].shape[0]

        original_idx = torch.arange(batch_size, device=self.device)
        new_idx = torch.randperm(batch_size, device=self.device)

        while not (original_idx == new_idx).sum().eq(0):
            new_idx = torch.randperm(batch_size, device=self.device)

        shuffled_imgs = self.databatch_set[tgt_idx]['imgs'][new_idx]
        shuffled_labels = [self.databatch_set[tgt_idx]['label'][i] for i in new_idx]

        return {
            'imgs': shuffled_imgs,
            'label': shuffled_labels,
        }

    def msg_set2msg_indices(self, msg_set:list) -> list:
        def _msgstring2indices_(msg):
            return [int(c) for c in msg]
        
        msg_indices = [_msgstring2indices_(msg) for msg in msg_set]
        
        return msg_indices

    def build_tensor_mask_lens_maxlen(self, indices_batch, value=args.pad_index):
        padded_indices = PairDataset.pad(indices_batch)
        
        lens = torch.tensor([len(indices) for indices in indices_batch]).to(self.device)
        max_len = max([len(indices) for indices in indices_batch])
        
        mask = PairDataset.build_mask(padded_indices, value)
        mask = torch.ByteTensor(mask).to(self.device)

        padded_indices = torch.LongTensor(padded_indices).to(self.device)
        
        return padded_indices, mask, lens, max_len

    def build_batches(self):
        batches = []

        img_names, imgs = ImgChooseDataset.load_img_set(self.dir_path)
        img_names = [name.split('.')[0] for name in img_names]

        lan_pairs = PairDataset.load_pairset(self.lan_path)
        language = {}
        for pair in lan_pairs:
            language[pair[0]] = pair[1]

        assert len(img_names) == len(imgs)
        c = list(zip(img_names, imgs))
        random.shuffle(c)
        img_names, imgs = zip(*c)
        img_names = list(img_names)
        imgs = list(imgs)

        msg_set = [language[name] for name in img_names]
        msg_indices = self.msg_set2msg_indices(msg_set)

        # ceil/floor
        if len(imgs) < self.batch_size:
            num_batches = 1
        else:
            num_batches = math.floor(len(imgs) / self.batch_size)

        for i in range(num_batches):
            img_batch = ImgChooseDataset.build_img_tensors(
                    imgs[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))],
                    device=self.device
                )
            img_label_batch = img_names[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))]
            msg_indices_batch = msg_indices[i*self.batch_size:min((i+1)*self.batch_size, len(imgs))]
            msg_var, msg_mask, msg_len, _ = \
                self.build_tensor_mask_lens_maxlen(msg_indices_batch, value=-1)

            batches.append({
                'imgs': img_batch,
                'label': img_label_batch,
                'message': msg_var,
                'msg_mask': msg_mask,
                'msg_lens': msg_len,
            })

        return batches


if __name__ == '__main__':
    dataset = ImgChooseDataset(
        batch_size=7, 
        dataset_dir_path='data/img_set_25/',
        device=torch.device('cpu'),
        d_num=11
    )

    print('correct batch:')
    print(dataset[0]['correct']['imgs'].shape)

    # print('first distract:')
    # print(dataset[0]['candidates'][0])
