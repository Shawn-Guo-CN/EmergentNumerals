from utils.conf import *


class Voc:
    def __init__(self):
        self.word2index = {'A': 1, 'B': 2, 'C': 3}
        self.index2word = {SOS_TOKEN: "SOS", EOS_TOKEN: "EOS",
                           1: 'A', 2: 'B', 3:'C'}
        self.num_words = 5  # Count SOS, EOS


def string2indices(voc, sequence):
    # here, I add SOS and EOS to the original string
    return [SOS_TOKEN] + [voc.word2index[word] for word in sequence] + [EOS_TOKEN]


def string_set2indices_set(voc, string_set):
    indices_set = []
    for str_pair in string_set:
        indices_set.append([string2indices(voc, str_pair[0]), string2indices(voc, str_pair[1])])
    return indices_set


def indices_set2data_batches(in_set, batch_size=BATCH_SIZE):
    input_indices = []
    target_indices = []

    for pair in in_set:
        # keep SOS and EOS
        input_indices.append(pair[0])
        target_indices.append(pair[1])

    input_batches = []
    target_batches = []
    num_batches = math.ceil(len(input_indices) / batch_size)
    for i in range(0, num_batches):
        input_batches.append(
           torch.LongTensor(
               np.asarray(input_indices[i*batch_size:min((i+1)*batch_size, 
               len(input_indices))]).transpose()).to(DEVICE)
        )
        target_batches.append(
            torch.LongTensor(
                np.asarray(target_indices[i*batch_size:min((i+1)*batch_size, 
                len(input_indices))]).transpose()).to(DEVICE)
        )
    
    return input_batches, target_batches


if __name__ == '__main__':
    voc = Voc()
    str_set = [['AAA', 'BBB'], ['BBB', 'CCC']]
    indices_set = string_set2indices_set(voc, str_set)
    input_batches, target_batches = indices_set2data_batches(indices_set, batch_size=2)
    print(input_batches)
    print(target_batches)
