from utils import *
import codecs
import os

import cPickle as pkl


class NER_DataLoader():
    def __init__(self, file_path):
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        self.file_path = file_path
        self.tag_vocab_path = os.path.join(file_path, ".tag_vocab")
        self.word_vocab_path = os.path.join(file_path, ".word_vocab")
        self.char_vocab_path = os.path.join(file_path, ".char_vocab")
        self.pos_vocab_path = os.path.join(file_path, ".pos_vocab")

        if os.path.exists(self.tag_vocab_path) and os.path.exists(self.word_vocab_path) and os.path.exists(self.char_vocab_path) and os.path.exists(self.pos_vocab_path):
            # TODO: encoding?
            self.tag_to_id = pkl.load(self.tag_vocab_path)
            self.word_to_id = pkl.load(self.word_vocab_path)
            self.char_to_id = pkl.load(self.char_vocab_path)
            self.pos_to_id = pkl.load(self.pos_vocab_path)
        else:
            self.tag_to_id, self.word_to_id, self.char_to_id, self.pos_to_id = self.read_file()
            self.word_to_id['<unk>'] = len(self.word_to_id)
            self.word_to_id['<eos>'] = 0
            self.char_to_id['<unk>'] = len(self.char_to_id)

            pkl.dump(self.tag_to_id, self.tag_vocab_path)
            pkl.dump(self.word_to_id, self.word_vocab_path)
            pkl.dump(self.char_to_id, self.char_vocab_path)
            pkl.dump(self.pos_to_id, self.pos_vocab_path)

        self.word_padding_token = 0

        # for char vocab and word vocab, we reserve id 0 for the eos padding, and len(vocab)-1 for the <unk>
        self.id_to_tag = {v: k for k, v in self.tag_to_id.iteritems()}
        self.id_to_word = {v: k for k, v in self.word_to_id.iteritems()}
        self.id_to_pos = {v: k for k, v in self.pos_to_id.iteritems()}
        self.id_to_char = {v: k for k, v in self.char_to_id.iteritems()}

        self.ner_vocab_size = len(self.id_to_tag)
        self.pos_vocab_size = len(self.id_to_pos)
        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def dict_add(self, vocab, item, shift=0):
        if item not in vocab:
            vocab[item] = len(vocab) + shift

    def read_one_line(self, line, tag_vocab, word_vocab, char_vocab, pos_vocab):
        for w in line:
            fields = w.split()
            word = fields[0]
            pos_tag = fields[1]
            ner_tag = fields[-1]
            for c in word:
                self.dict_add(char_vocab, c, 1)
            self.dict_add(tag_vocab, ner_tag)
            self.dict_add(word_vocab, word, 1)
            self.dict_add(pos_vocab, pos_tag)

    def read_file(self):
        tag_vocab = {}
        word_vocab = {}
        char_vocab = {}
        pos_vocab = {}

        with codecs.open(self.file_path, "r", "utf-8") as fin:
            to_read_line = []
            for line in fin:
                if line.strip() == "":
                    self.read_one_line(to_read_line, tag_vocab, word_vocab, char_vocab, pos_vocab)
                else:
                    to_read_line.append(line.strip())
        return tag_vocab, word_vocab, char_vocab, pos_vocab

    def get_data_set(self):
        sents = []
        char_sents = []
        tgt_tags = []
        pos_tags = []

        with codecs.open(self.file_path, "r", "utf-8") as fin:
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    temp_sent = []
                    temp_pos = []
                    temp_ner = []
                    temp_char = []
                    for w in one_sent:
                        fields = w.split()
                        word = fields[0]
                        pos_tag = fields[1]
                        ner_tag = fields[-1]
                        temp_sent.append(self.word_to_id[word] if word in self.word_to_id else 0)
                        temp_pos.append(self.pos_to_id[pos_tag])
                        temp_ner.append(self.tag_to_id[ner_tag])
                        temp_char.append([self.char_to_id[c] if c in self.char_to_id else 0 for c in word])
                    sents.append(temp_sent)
                    char_sents.append(temp_char)
                    tgt_tags.append(temp_ner)
                    pos_tags.append(temp_pos)
                else:
                    one_sent.append(line.strip())

        return sents, char_sents, tgt_tags, pos_tags


class NER_DataLoader_No_Pos():
    def __init__(self, file_path, pretrained_embedding_path=None, use_discrete_feature=False):
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        self.file_path = file_path
        self.tag_vocab_path = os.path.join(file_path, ".tag_vocab")
        self.word_vocab_path = os.path.join(file_path, ".word_vocab")
        self.char_vocab_path = os.path.join(file_path, ".char_vocab")

        self.use_discrete_feature = use_discrete_feature

        if os.path.exists(self.tag_vocab_path) and os.path.exists(self.word_vocab_path) and os.path.exists(self.char_vocab_path) and os.path.exists(self.pos_vocab_path):
            # TODO: encoding?
            self.tag_to_id = pkl.load(self.tag_vocab_path)
            self.word_to_id = pkl.load(self.word_vocab_path)
            self.char_to_id = pkl.load(self.char_vocab_path)
        else:
            self.tag_to_id, self.word_to_id, self.char_to_id = self.read_file()
            self.word_to_id['<unk>'] = len(self.word_to_id)
            self.word_to_id['<eos>'] = 0
            self.char_to_id['<unk>'] = len(self.char_to_id)

            pkl.dump(self.tag_to_id, self.tag_vocab_path)
            pkl.dump(self.word_to_id, self.word_vocab_path)
            pkl.dump(self.char_to_id, self.char_vocab_path)

        self.word_padding_token = 0

        if pretrained_embedding_path is not None:
            self.pretrain_word_emb, self.word_to_id = get_pretrained_emb(pretrained_embedding_path)

        # for char vocab and word vocab, we reserve id 0 for the eos padding, and len(vocab)-1 for the <unk>
        self.id_to_tag = {v: k for k, v in self.tag_to_id.iteritems()}
        self.id_to_word = {v: k for k, v in self.word_to_id.iteritems()}
        self.id_to_char = {v: k for k, v in self.char_to_id.iteritems()}

        self.ner_vocab_size = len(self.id_to_tag)
        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def dict_add(self, vocab, item, shift=0):
        if item not in vocab:
            vocab[item] = len(vocab) + shift

    def read_one_line(self, line, tag_vocab, word_vocab, char_vocab):
        for w in line:
            fields = w.split()
            word = fields[0]
            ner_tag = fields[-1]
            for c in word:
                self.dict_add(char_vocab, c, 1)
            self.dict_add(tag_vocab, ner_tag)
            self.dict_add(word_vocab, word, 1)

    def read_file(self):
        tag_vocab = {}
        word_vocab = {}
        char_vocab = {}

        with codecs.open(self.file_path, "r", "utf-8") as fin:
            to_read_line = []
            for line in fin:
                if line.strip() == "":
                    self.read_one_line(to_read_line, tag_vocab, word_vocab, char_vocab)
                else:
                    to_read_line.append(line.strip())
        return tag_vocab, word_vocab, char_vocab

    def get_data_set(self):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []

        with codecs.open(self.file_path, "r", "utf-8") as fin:
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    temp_sent = []
                    temp_ner = []
                    temp_char = []
                    temp_discrete = []
                    for w in one_sent:
                        fields = w.split()
                        word = fields[0]
                        ner_tag = fields[-1]
                        if self.use_discrete_feature:
                            temp_discrete.append(get_feature_w(word))
                        temp_sent.append(self.word_to_id[word] if word in self.word_to_id else 0)
                        temp_ner.append(self.tag_to_id[ner_tag])
                        temp_char.append([self.char_to_id[c] if c in self.char_to_id else 0 for c in word])
                    sents.append(temp_sent)
                    char_sents.append(temp_char)
                    tgt_tags.append(temp_ner)
                    discrete_features.append(temp_discrete)
                else:
                    one_sent.append(line.strip())
        self.num_feats = len(discrete_features[0][0])
        return sents, char_sents, tgt_tags, discrete_features