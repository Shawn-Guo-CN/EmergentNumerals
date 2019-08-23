import torch
import matplotlib.pyplot as plt
from math import inf
import numpy as np
import os

from utils.conf import args


def load_training_status(file_path:str) -> tuple:
    print('loading '+file_path+'...')
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    records = checkpoint['records']
    print(checkpoint['args'])
    print('done')
    return records


def EMA_smooth(l:list, alpha=0.05) -> list:
    if len(l) == 0:
        return l
    new_l = [l[0]]
    for i in l:
        new_l.append(alpha * i + (1-alpha) * new_l[-1])
    return new_l


def load_saved_param_as_2d_array(dir:str):
    losses = []
    tok_accs = []
    seq_accs = []

    files = os.listdir(dir)
    for file_name in files:
        records = load_training_status(os.path.join(dir, file_name))
        seq_accs.append(EMA_smooth(records['training_seq_acc'][:400]))
        tok_accs.append(EMA_smooth(records['training_tok_acc'][:400]))
        losses.append(EMA_smooth(records['training_loss'][:400]))

    losses = np.asarray(losses)
    tok_accs = np.asarray(tok_accs)
    seq_accs = np.asarray(seq_accs)

    return losses, seq_accs, tok_accs


def get_lines4plotting(data):
    mean = np.mean(data, axis=0)
    var = np.std(data, axis=0)
    upper_line = mean + var
    bottom_line = mean - var

    return mean.tolist(), upper_line.tolist(), bottom_line.tolist()


def plot_into_1_figure(chk_points_dirs:list, label_list:list) -> None:

    fig, ((ax0, ax1, ax2)) = plt.subplots(nrows=1, ncols=3)

    for idx, dir in enumerate(chk_points_dirs):
        loss, s_acc, t_acc = load_saved_param_as_2d_array(dir)

        l_mean, l_up, l_bottom = get_lines4plotting(loss)
        s_mean, s_up, s_bottom = get_lines4plotting(s_acc)
        t_mean, t_up, t_bottom = get_lines4plotting(t_acc)

        linewidth = 1.0

        ax0.plot(l_mean, label=label_list[idx], linewidth=linewidth)
        ax0.fill_between(range(len(l_mean)), l_up, l_bottom, alpha=0.3)

        ax1.plot(s_mean, label=label_list[idx], linewidth=linewidth)
        ax1.fill_between(range(len(s_mean)), s_up, s_bottom, alpha=0.3)

        ax2.plot(t_mean, label=label_list[idx], linewidth=linewidth)
        ax2.fill_between(range(len(t_mean)), t_up, t_bottom, alpha=0.3)
        
    
    ax0.legend()
    ax0.set_xlabel('Number of Epochs')
    ax0.title.set_text('Training Loss')

    ax1.legend()
    ax1.set_xlabel('Number of Epochs')
    ax1.title.set_text('Training Sequence Accuracy')

    ax2.legend()
    ax2.set_xlabel('Number of Epochs')
    ax2.title.set_text('Training Token Accuracy')
    plt.show()

def main():
    chk_point_dir_path_list = [
        './params/test_learning_speed_0813/speaker/comp_4/',
        './params/test_learning_speed_0813/speaker/holistic_4/',
        './params/test_learning_speed_0813/speaker/emergent_4_gen/',
        './params/test_learning_speed_0813/speaker/emergent_2/',
        './params/test_learning_speed_0813/speaker/comp_2/',
        './params/test_learning_speed_0813/speaker/holistic_2/',
        # './params/test_learning_speed_0813/speaker/emergent_4_refer/',
        './params/test_learning_speed_0813/speaker/set2seq/',
    ]

    label_list = [
        'compositional (len 4)',
        'holistic (len 4)',
        'emergent - reconstruct (len 4)',
        'emergent - select (len 2)',
        'compositional (len 2)',
        'holistic (len 2)',
        # 'emergent - select (len 4)',
        'set2seq',
    ]

    plot_into_1_figure(chk_point_dir_path_list, label_list)


if __name__ == '__main__':
    main()
