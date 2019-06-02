from utils.conf import *


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