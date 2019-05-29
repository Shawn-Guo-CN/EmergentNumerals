from utils.conf import *
# for testing modules of standard Seq2Seq model
from models.Seq2Seq import *


class Voc:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}

        self.word2index[PAD_TOKEN] = PAD_INDEX
        self.index2word[PAD_INDEX] = PAD_TOKEN
        self.word2index[SOS_TOKEN] = SOS_INDEX
        self.index2word[SOS_INDEX] = SOS_TOKEN
        self.word2index[EOS_TOKEN] = EOS_INDEX
        self.index2word[EOS_INDEX] = EOS_TOKEN
        for i in range(1, NUM_WORD+1):
            self.word2index[chr(64+i)] = EOS_INDEX + i
            self.index2word[EOS_INDEX + i] = chr(64+i)
        
        self.num_words = NUM_WORD + 3  # include SOS, EOS and PAD


def string2indices(voc, sequence):
    return [SOS_INDEX] + [voc.word2index[word] for word in sequence] + [EOS_INDEX]


def string_set2indices_pair_set(voc, string_set):
    indices_pair_set = []
    for string in string_set:
        # input contains neither SOS or EOS, target contains both SOS and EOS
        indices_pair_set.append([string2indices(voc, string)[1:-1], string2indices(voc, string)])
    return indices_pair_set


def pad(l, fillvalue=PAD_INDEX):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l, value=PAD_INDEX):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for index in seq:
            if index == PAD_INDEX:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def indices_pair_set2data_batches(in_set, batch_size=BATCH_SIZE):
    input_indices = []
    target_indices = []

    for pair in in_set:
        input_indices.append(pair[0])
        target_indices.append(pair[1])

    def _input_var_(input_indices):
        lengths = torch.tensor([len(indexes) for indexes in input_indices])
        paadded_input = pad(input_indices)
        padded_input = torch.LongTensor(paadded_input).to(DEVICE)
        return padded_input, lengths
    
    def _binary_matrix_(l, value=PAD_INDEX):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_INDEX:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def _target_var_(target_indices):
        max_target_len = max([len(indexes) for indexes in target_indices])
        padded_target = pad(target_indices)
        mask = _binary_matrix_(padded_target)
        mask = torch.ByteTensor(mask).to(DEVICE)
        padded_target = torch.LongTensor(padded_target).to(DEVICE)
        return padded_target, mask, max_target_len

    input_batches = []
    input_length_batches = []
    target_batches = []
    target_mask_batches = []
    target_max_len_batches = []

    num_batches = math.ceil(len(input_indices) / batch_size)
    for i in range(0, num_batches):
        input_indices_batch = input_indices[i*batch_size:min((i+1)*batch_size, len(input_indices))]
        target_indices_batch = target_indices[i*batch_size:min((i+1)*batch_size, len(input_indices))]
        input_indices_batch.sort(key=len, reverse=True)
        target_indices_batch.sort(key=len, reverse=True)

        input_var, input_lengths = _input_var_(input_indices_batch)
        input_batches.append(input_var)
        input_length_batches.append(input_lengths)
        
        target_var, target_mask, target_max_len = _target_var_(target_indices_batch)
        
        target_batches.append(target_var)
        target_mask_batches.append(target_mask)
        target_max_len_batches.append(target_max_len)

    return input_batches, input_length_batches, target_batches, \
                target_mask_batches, target_max_len_batches


if __name__ == '__main__':
    voc = Voc()
    print(voc.word2index)
    print(voc.index2word)
    
    str_set = ['ABCDEFF', 'ABBCDEEF']
    indices_pair_set = string_set2indices_pair_set(voc, str_set)
    inputs, input_lens, targets, target_masks, target_lens = \
        indices_pair_set2data_batches(indices_pair_set, batch_size=2)

    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
    encoder = EncoderLSTM(embedding)

    for i in range(0, len(inputs)):
        # print('input:', inputs[i])
        # print('input lengths:', input_lens[i])
        # print('targets:', targets[i])
        # print('target mask:', target_masks[i])
        # print('target max len:', target_lens[i])
        encoder_outputs, encoder_hidden = encoder(inputs[i], input_lens[i])
        print(encoder_hidden.shape)
    
