import torch
import numpy as np

from utils.conf import args
from models.Set2Seq2Seq import Set2Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc

DATA_FILE = './data/all_data.txt'
OUT_FILE = './data/input_msg_pairs.txt'

def print_input_message_pair(input_str, message_tensor, mask_tensor, out_file):
    print('---', file=out_file)
    print(input_str, file=out_file)
    message = message_tensor.squeeze().detach().cpu().numpy()
    # print(message[0], file=out_file)
    max_idx = []
    for r_idx in range(message.shape[0]):
        max_idx.append(np.argmax(message[r_idx]))
    print(max_idx, file=out_file)


def iterate_dataset(model, str_set, batch_set, out_file):
    for idx, data_batch in enumerate(batch_set):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        speaker_input = model.embedding(input_var.t())
        message, msg_mask, _ = model.speaker(speaker_input, input_mask)
        print_input_message_pair(str_set[idx], message, msg_mask, out_file)


def main():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    data_set = FruitSeqDataset(voc, dataset_file_path=DATA_FILE, batch_size=1)
    str_set = data_set.load_stringset(DATA_FILE)
    print('done')

    print('rebuilding model from saved parameters in ' + args.param_file + '...')
    model = Set2Seq2Seq(voc.num_words).to(args.device)
    checkpoint = torch.load(args.param_file)
    args = checkpoint['args']
    model.load_state_dict(checkpoint['model'])
    voc = checkpoint['voc']
    print('done')

    model.eval()

    print('iterating data set...')
    dev_out_file = open(OUT_FILE, mode='a')
    iterate_dataset(model, str_set, data_set, dev_out_file)


if __name__ == '__main__':
    main()
