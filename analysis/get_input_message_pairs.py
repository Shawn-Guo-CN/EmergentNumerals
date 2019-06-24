import torch
import numpy as np

from utils.conf import args
from models.Set2Seq2Seq import Set2Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc

DATA_FILE = './data/all_data.txt'
OUT_FILE = './data/input_msg_pairs.txt'


def reproduce_msg_output(model, voc, data_batch, train_args):
    input_var = data_batch['input']
    input_mask = data_batch['input_mask']
    target_var = data_batch['target']
    target_mask = data_batch['target_mask']
    target_max_len = data_batch['target_max_len']
    speaker_input = model.embedding(input_var.t())
    message, msg_mask, _ = model.speaker(speaker_input, input_mask)
    output = model.listener(model.embedding, message, msg_mask, 
                        target_var, target_mask, target_max_len)[-1]
    
    message = message.squeeze().detach().cpu().numpy()
    msg_str = ''
    msg_end = False
    for r_idx in range(message.shape[0]):
        cur_v = np.argmax(message[r_idx])
        if cur_v == train_args.msg_vocsize - 1:
            msg_end = True
        if not msg_end:
            msg_str += str(cur_v)
    msg_str += '/'

    output = output.squeeze().detach().cpu().numpy()
    output_str = ''
    output_end = False
    for r_idx in range(output.shape[0]):
        cur_v = np.argmax(output[r_idx])
        if cur_v == train_args.eos_index:
            output_end = True
        if not output_end:
            output_str += voc.index2word[cur_v]
    output_str += '/'

    return msg_str, output_str


def iterate_dataset(model, voc, str_set, batch_set, out_file, train_args):
    for idx, data_batch in enumerate(batch_set):
        message, output = reproduce_msg_output(model, voc, data_batch, train_args)
        print(str_set[idx] + '\t' + message + '\t' + output, file=out_file)


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
    checkpoint = torch.load(args.param_file, map_location=args.device)
    train_args = checkpoint['args']
    model.load_state_dict(checkpoint['model'])
    voc = checkpoint['voc']
    print('done')

    model.eval()

    print('iterating data set...')
    out_file = open(OUT_FILE, mode='a')
    iterate_dataset(model, voc, str_set, data_set, out_file, train_args)


if __name__ == '__main__':
    main()
