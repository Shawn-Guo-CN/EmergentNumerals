import torch
import numpy as np

from utils.conf import args
from models.Set2Seq2Seq import Set2Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


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
    dev_set = FruitSeqDataset(voc, dataset_file_path=args.dev_file, batch_size=1)
    dev_str_set = dev_set.load_stringset(args.dev_file)
    test_set = FruitSeqDataset(voc, dataset_file_path=args.test_file, batch_size=1)
    test_str_set = test_set.load_stringset(args.test_file)
    print('done')

    print('rebuilding model from saved parameters in ' + args.param_file + '...')
    model = Set2Seq2Seq(voc.num_words).to(args.device)
    checkpoint = torch.load(args.param_file)
    model.load_state_dict(checkpoint['model'])
    voc = checkpoint['voc']
    print('done')

    model.eval()

    print('iterating dev set...')
    dev_out_file = open('./data/dev_messages.txt', mode='a')
    iterate_dataset(model, dev_str_set, dev_set, dev_out_file)

    print('iterating test set...')
    test_out_file = open('./data/test_messages.txt', mode='a')
    iterate_dataset(model, test_str_set, test_set, test_out_file)



if __name__ == '__main__':
    main()
