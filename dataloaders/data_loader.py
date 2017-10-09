__author__ = 'chuntingzhou'
from models.utils import *
import codecs
import os


class NER_DataLoader():
    def __init__(self, args, pretrained_embedding_path=None, use_discrete_feature=False):
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        self.train_path = args.train_path

        self.tag_vocab_path = os.path.join(self.train_path, ".tag_vocab")
        self.word_vocab_path = os.path.join(self.train_path, ".word_vocab")
        self.char_vocab_path = os.path.join(self.train_path, ".char_vocab")

        self.use_discrete_feature = use_discrete_feature

        if os.path.exists(self.tag_vocab_path) and os.path.exists(self.word_vocab_path) and os.path.exists(self.char_vocab_path):
            # TODO: encoding?
            self.tag_to_id = pkl_load(self.tag_vocab_path)
            self.word_to_id = pkl_load(self.word_vocab_path)
            self.char_to_id = pkl_load(self.char_vocab_path)
        else:
            self.tag_to_id, self.word_to_id, self.char_to_id = self.read_file()
            self.word_to_id['<unk>'] = len(self.word_to_id)
            self.word_to_id['<eos>'] = 0
            self.char_to_id['<unk>'] = len(self.char_to_id)

            pkl_dump(self.tag_to_id, self.tag_vocab_path)
            pkl_dump(self.char_to_id, self.char_vocab_path)
            pkl_dump(self.word_to_id, self.word_vocab_path)

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

        print("NER tag num=%d, Word vocab size=%d, Char Vocab size=%d" % (self.ner_vocab_size, self.word_vocab_size, self.char_vocab_size))

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def dict_add(self, vocab, item, shift=0):
        if item not in vocab:
            vocab[item] = len(vocab) + shift

    def read_one_line(self, line, tag_list, word_list, char_list):
        for w in line:
            fields = w.split()
            word = fields[0]
            ner_tag = fields[-1]
            for c in word:
                char_list.append(c)
            tag_list.append(ner_tag)
            word_list.append(word)

    def read_one_line_set(self, line, tag_set, word_set, char_set):
        for w in line:
            fields = w.split()
            word = fields[0]
            ner_tag = fields[-1]
            for c in word:
                char_set.add(c)
            tag_set.add(ner_tag)
            word_set.add(ner_tag)

    def get_vocab(self, a_set, shift=0):
        vocab = {}
        for i, elem in enumerate(a_set):
            vocab[elem] = i + shift

        return vocab

    def read_file(self):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_set = set()
        char_set = set()
        tag_set = set()
        with codecs.open(self.train_path, "r", "utf-8") as fin:
            to_read_line = []
            num_lines = 0
            for line in fin:
                num_lines += 1
                if num_lines % 100 == 0:
                    print("processed %d lines. " % num_lines)
                if line.strip() == "":
                    self.read_one_line_set(to_read_line, tag_set, word_set, char_set)
                else:
                    to_read_line.append(line.strip())
        # tag_set = set(tag_list)
        # word_set = set(tag_list)
        # char_set = set(char_list)

        tag_vocab = self.get_vocab(tag_set)
        word_vocab = self.get_vocab(word_set, 1)
        char_vocab = self.get_vocab(char_set)

        return tag_vocab, word_vocab, char_vocab

    def get_data_set(self, path):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []

        with codecs.open(path, "r", "utf-8") as fin:
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