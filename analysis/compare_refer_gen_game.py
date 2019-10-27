import torch
import matplotlib.pyplot as plt
from math import inf
import numpy as np

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


def plot_training_into_1_figure(chk_points_paths:list, label_list:list) -> None:
    for idx, file_path in enumerate(chk_points_paths):
        records = load_training_status(file_path)
        t_sim = records['training_spkh_lish_sim'][:] # records['training_sim']

        t_sim   = EMA_smooth(t_sim)

        linewidth = 1.0

        plt.plot(t_sim, label=label_list[idx], linewidth=linewidth)
        plt.legend()

    plt.xlabel('Number of Iterations')
    plt.title('Topological Similarity Between Speaker Hidden and Listener Hidden')
    plt.yticks(np.arange(0, 1, step=0.05))
    plt.grid()
    plt.show()

def main():
    chk_point_path_list = [
        './params/compare_spkh_lish_sim/gen_game.tar',
        './params/compare_spkh_lish_sim/refer_game.tar'
    ]

    label_list = [
        'referential game',
        'reconstruction game'
    ]

    plot_training_into_1_figure(chk_point_path_list, label_list)


if __name__ == '__main__':
    main()
