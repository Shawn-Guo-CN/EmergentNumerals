import pandas as pd
from analysis.get_input_message_pairs import generate_in_msg_pairs

DATA_FILE = './data/all_data.txt'


def main():
    ins, msgs = generate_in_msg_pairs(DATA_FILE)
    for im in zip(ins, msgs):
        print(im)


if __name__ == '__main__':
    main()