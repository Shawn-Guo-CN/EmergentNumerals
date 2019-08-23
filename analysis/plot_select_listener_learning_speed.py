import torch
import matplotlib.pyplot as plt
from math import inf
import numpy as np
import os
from analysis.plot_gen_listener_learning_speed import load_training_status, EMA_smooth

from utils.conf import args


def load_saved_param_as_2d_array(dir:str):
    losses = []
    train_accs = []
    eval_accs = []

    files = os.listdir(dir)
    for file_name in files:
        records = load_training_status(os.path.join(dir, file_name))
        train_accs.append(EMA_smooth(records['training_acc'][:400], alpha=0.1))
        # eval_accs.append(EMA_smooth(records['eval_acc'][:400]))
        losses.append(EMA_smooth(records['training_loss'][:400], alpha=0.1))

    losses = np.asarray(losses)
    train_accs = np.asarray(train_accs)
    #eval_accs = np.asarray(eval_accs)

    return losses, train_accs # , eval_accs


def get_lines4plotting(data):
    mean = np.mean(data, axis=0)
    var = np.std(data, axis=0)
    upper_line = mean + var
    bottom_line = mean - var

    return mean.tolist(), upper_line.tolist(), bottom_line.tolist()


def plot_into_1_figure(chk_points_dirs:list, label_list:list) -> None:

    fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2)

    for idx, dir in enumerate(chk_points_dirs):
        loss, t_acc = load_saved_param_as_2d_array(dir)

        l_mean, l_up, l_bottom = get_lines4plotting(loss)
        t_mean, t_up, t_bottom = get_lines4plotting(t_acc)

        linewidth = 1.0

        ax0.plot(l_mean, label=label_list[idx], linewidth=linewidth)
        ax0.fill_between(range(len(l_mean)), l_up, l_bottom, alpha=0.3)

        ax1.plot(t_mean, label=label_list[idx], linewidth=linewidth)
        ax1.fill_between(range(len(t_mean)), t_up, t_bottom, alpha=0.3)
        
    
    ax0.legend()
    ax0.set_xlabel('Number of Epochs')
    ax0.title.set_text('Training Loss')

    ax1.legend()
    ax1.set_xlabel('Number of Epochs')
    ax1.title.set_text('Training Accuracy')

    plt.show()

def main():
    chk_point_dir_path_list = [
        './params/test_learning_speed_0813/choose_listener/comp_4/',
        './params/test_learning_speed_0813/choose_listener/holistic_4/',
        './params/test_learning_speed_0813/choose_listener/emergent_4/',
        './params/test_learning_speed_0813/choose_listener/emergent_2/',
        './params/test_learning_speed_0813/choose_listener/comp_2/',
        './params/test_learning_speed_0813/choose_listener/holistic_2/',
    ]

    label_list = [
        'compositional (len 4)',
        'holistic (len 4)',
        'emergent (len 4)',
        'emergent (len 2)',
        'compositional (len 2)',
        'holistic (len 2)',
    ]

    plot_into_1_figure(chk_point_dir_path_list, label_list)


if __name__ == '__main__':
    main()
