import torch
import matplotlib.pyplot as plt
from math import inf
import numpy as np
import os
from analysis.plot_gen_listener_learning_speed import load_training_status, EMA_smooth, get_lines4plotting

from utils.conf import args


def load_saved_param_as_2d_array(dir:str):
    losses = []
    t_s_acc = []
    t_t_acc = []
    e_s_acc = []
    e_t_acc = []

    files = os.listdir(dir)
    for file_name in files:
        records = load_training_status(os.path.join(dir, file_name))
        t_s_acc.append(EMA_smooth(records['training_seq_acc'][:400], alpha=0.1))
        t_t_acc.append(EMA_smooth(records['training_tok_acc'][:400], alpha=0.1))
        e_s_acc.append(EMA_smooth(records['eval_seq_acc'][:400], alpha=0.1))
        e_t_acc.append(EMA_smooth(records['eval_tok_acc'][:400], alpha=0.1))
        losses.append(EMA_smooth(records['training_loss'][:400], alpha=0.1))

    losses = np.asarray(losses)
    t_s_acc = np.asarray(t_s_acc)
    t_t_acc = np.asarray(t_t_acc)
    e_s_acc = np.asarray(e_s_acc)
    e_t_acc = np.asarray(e_t_acc)

    return losses, t_s_acc, t_t_acc, e_s_acc, e_t_acc


def plot_into_1_figure(chk_points_dirs:list, label_list:list) -> None:

    fig = plt.figure()
    ax0 = fig.add_subplot(231)
    ax1 = fig.add_subplot(234)
    ax2 = fig.add_subplot(233)
    ax3 = fig.add_subplot(236)
    ax4 = fig.add_subplot(132)


    for idx, dir in enumerate(chk_points_dirs):
        loss, t_s_acc, t_t_acc, e_s_acc, e_t_acc = load_saved_param_as_2d_array(dir)

        l_mean, l_up, l_bottom = get_lines4plotting(loss)
        t_s_mean, t_s_up, t_s_bottom = get_lines4plotting(t_s_acc)
        t_t_mean, t_t_up, t_t_bottom = get_lines4plotting(t_t_acc)
        e_s_mean, e_s_up, e_s_bottom = get_lines4plotting(e_s_acc)
        e_t_mean, e_t_up, e_t_bottom = get_lines4plotting(e_t_acc)

        linewidth = 1.0

        ax0.plot(e_s_mean, label=label_list[idx], linewidth=linewidth)
        ax0.fill_between(range(len(e_s_mean)), e_s_up, e_s_bottom, alpha=0.3)

        ax1.plot(e_t_mean, label=label_list[idx], linewidth=linewidth)
        ax1.fill_between(range(len(e_t_mean)), e_t_up, e_t_bottom, alpha=0.3)

        ax4.plot(l_mean, label=label_list[idx], linewidth=linewidth)
        ax4.fill_between(range(len(l_mean)), l_up, l_bottom, alpha=0.3)
        
        ax2.plot(t_s_mean, label=label_list[idx], linewidth=linewidth)
        ax2.fill_between(range(len(t_s_mean)), t_s_up, t_s_bottom, alpha=0.3)

        ax3.plot(t_t_mean, label=label_list[idx], linewidth=linewidth)
        ax3.fill_between(range(len(t_t_mean)), t_t_up, t_t_bottom, alpha=0.3)

        
    
    ax0.legend(loc=4)
    ax0.set_xlabel('Number of Epochs')
    ax0.title.set_text('Evaluation Sequence Accuracy')

    ax1.legend(loc=4)
    ax1.set_xlabel('Number of Epochs')
    ax1.title.set_text('Evaluation Token Accuracy')

    ax2.legend(loc=4)
    ax2.set_xlabel('Number of Epochs')
    ax2.title.set_text('Training Sequence Accuracy')

    ax3.legend(loc=4)
    ax3.set_xlabel('Number of Epochs')
    ax3.title.set_text('Training Token Accuracy')

    ax4.legend()
    ax4.set_xlabel('Number of Epochs')
    ax4.title.set_text('Training Loss')
    
    plt.subplots_adjust(hspace=0.2)
    plt.show()

def main():
    chk_point_dir_path_list = [
        './params/test_language_generalise_0813/generate/comp_4/',
        './params/test_language_generalise_0813/generate/emergent_gen_8/',
        './params/test_language_generalise_0813/generate/emergent_refer_8/',
        './params/test_language_generalise_0813/generate/holistic_4/',
        './params/test_language_generalise_0813/generate/seq2seq/',
    ]

    label_list = [
        'compositional (len 4)',
        'emergent - reconstruct (len 8)',
        'emergent - select (len 8)',
        'holistic (len 4)',
        'seq2seq',
    ]

    plot_into_1_figure(chk_point_dir_path_list, label_list)


if __name__ == '__main__':
    main()
