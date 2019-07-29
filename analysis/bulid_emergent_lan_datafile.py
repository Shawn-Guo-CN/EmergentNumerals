import torch

# from models.Set2Seq2Seq import Set2Seq2Seq
from models.Set2Seq2Choice import Set2Seq2Choice
from utils.conf import args
from analysis.training_sim_check import reproduce_msg_set
from preprocesses.DataIterator import FruitSeqDataset, ChooseDataset


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


if __name__ == '__main__':
    if args.param_file is not None:
        checkpoint = torch.load(args.param_file, map_location=torch.device('cpu'))
    else:
        raise ValueError

    print('rebuilding vocabulary and model...')
    voc = checkpoint['voc']
    train_args = checkpoint['args']
    print(train_args)
    # model = Set2Seq2Seq(voc.num_words).to(torch.device('cpu'))
    model = Set2Seq2Choice(voc.num_words, msg_length=train_args.max_msg_len, msg_vocsize=train_args.msg_vocsize, 
                    hidden_size=train_args.hidden_size, dropout=train_args.dropout_ratio, msg_mode=train_args.msg_mode
                    ).to(torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('done')
    
    print('loading and building batch dataset...')
    # batch_set = FruitSeqDataset(voc, dataset_file_path=args.data_file, batch_size=1, device=torch.device('cpu'))
    batch_set = ChooseDataset(voc, dataset_file_path=args.data_file, batch_size=1, device=torch.device('cpu'))
    in_set = FruitSeqDataset.load_stringset(args.data_file)
    print('done')

    build_listener_training_file(model, in_set, batch_set, 'data/rebuilt_language_2.txt')
