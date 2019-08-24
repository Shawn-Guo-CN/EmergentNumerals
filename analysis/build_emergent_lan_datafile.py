import torch

from models.Set2Seq2Seq import Set2Seq2Seq
from models.Set2Seq2Choice import Set2Seq2Choice
from models.Img2Seq2Choice import Img2Seq2Choice
from utils.conf import args
from analysis.training_sim_check import reproduce_msg_set
from preprocesses.DataIterator import FruitSeqDataset, ChooseDataset, ImgChooseDataset


def msg2str(msg):
    return ''.join([str(c) for c in msg])


def build_listener_training_file(model, in_set, batch_set, file_path='data/rebuilt_language.txt'):
    model.eval()
    msg_set = reproduce_msg_set(model, batch_set)

    assert len(in_set) == len(batch_set)

    out_file = open(file_path, 'a')
    for i in range(len(in_set)):
        print(msg2str(msg_set[i]) + '\t' + in_set[i], file=out_file)

    out_file.close()


def main(
    model_name='Img2Seq2Choice',
    dataset_name='ImgChooseDataset',
    out_file_path='data/tmp.txt',
):
    if args.param_file is not None:
        checkpoint = torch.load(args.param_file, map_location=torch.device('cpu'))
    else:
        raise ValueError

    print('rebuilding vocabulary and model...')
    voc = checkpoint['voc'] if model_name == 'Set2Seq2Seq' or model_name == 'Set2Seq2Choice' else None
    train_args = checkpoint['args']
    print(train_args)

    if model_name == 'Img2Seq2Choice':
        model = Img2Seq2Choice(
                msg_length=train_args.max_msg_len, msg_vocsize=train_args.msg_vocsize,
                hidden_size=train_args.hidden_size, dropout=train_args.dropout_ratio, msg_mode=train_args.msg_mode
            ).to(torch.device('cpu'))
    elif model_name == 'Set2Seq2Seq':
        model = Set2Seq2Seq(
                voc.num_words, msg_length=train_args.max_msg_len, msg_vocsize=train_args.msg_vocsize, 
                hidden_size=train_args.hidden_size, dropout=train_args.dropout_ratio, msg_mode=train_args.msg_mode
            ).to(torch.device('cpu'))
    elif model_name == 'Set2Seq2Choice':
        model = Set2Seq2Choice(
                voc.num_words, msg_length=train_args.max_msg_len, msg_vocsize=train_args.msg_vocsize, 
                hidden_size=train_args.hidden_size, dropout=train_args.dropout_ratio, msg_mode=train_args.msg_mode
            ).to(torch.device('cpu'))
    else:
        raise NotImplementedError
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('done')
    
    print('loading and building batch dataset...')
    if dataset_name == 'ImgChooseDataset':
        batch_set = ImgChooseDataset(dataset_dir_path=args.data_file, batch_size=1, device=torch.device('cpu'))
        in_set = [batch['correct']['label'][0] for batch in batch_set]
    elif dataset_name == 'FruitSeqDataset':
        batch_set = FruitSeqDataset(voc, dataset_file_path=args.data_file, batch_size=1, device=torch.device('cpu'))
        in_set = FruitSeqDataset.load_stringset(args.data_file)
    elif dataset_name == 'ChooseDataset':
        batch_set = ChooseDataset(voc, dataset_file_path=args.data_file, batch_size=1, device=torch.device('cpu'))
        in_set = FruitSeqDataset.load_stringset(args.data_file)
    print('done')

    build_listener_training_file(model, in_set, batch_set, out_file_path)


if __name__ == '__main__':
    main(
        model_name='Img2Seq2Choice',
        dataset_name='ImgChooseDataset',
        out_file_path='./data/img_languages/emergent/test.txt',
    )
