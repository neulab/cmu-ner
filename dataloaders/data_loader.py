__author__ = 'chuntingzhou'
from models.utils import *
import codecs
import os
from models.features import *

class NER_DataLoader():
    def __init__(self, args):
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        self.train_path = args.train_path
        self.args = args

        self.tag_vocab_path = self.train_path + ".tag_vocab"
        self.word_vocab_path = self.train_path + ".word_vocab"
        self.char_vocab_path = self.train_path + ".char_vocab"

        self.pretrained_embedding_path = args.pretrain_emb_path
        self.use_discrete_feature = args.use_discrete_features

        if os.path.exists(self.tag_vocab_path) and os.path.exists(self.word_vocab_path) and os.path.exists(self.char_vocab_path):
            # TODO: encoding?
            print("Load vocabs from file ....")
            self.tag_to_id = pkl_load(self.tag_vocab_path)
            self.word_to_id = pkl_load(self.word_vocab_path)
            self.char_to_id = pkl_load(self.char_vocab_path)
            print("Done!")
        else:
            self.tag_to_id, self.word_to_id, self.char_to_id = self.read_file()
            self.word_to_id['<eos>'] = 0
            self.char_to_id['<pad>'] = 0
            self.word_to_id['<unk>'] = len(self.word_to_id)
            self.char_to_id['<unk>'] = len(self.char_to_id)

            pkl_dump(self.tag_to_id, self.tag_vocab_path)
            pkl_dump(self.char_to_id, self.char_vocab_path)
            pkl_dump(self.word_to_id, self.word_vocab_path)

        self.word_padding_token = 0
        self.char_padding_token = 0

        if self.pretrained_embedding_path is not None:
            self.pretrain_word_emb, self.word_to_id = get_pretrained_emb(self.pretrained_embedding_path,
                                                                         self.word_to_id, args.word_emb_dim)
        # for char vocab and word vocab, we reserve id 0 for the eos padding, and len(vocab)-1 for the <unk>
        self.id_to_tag = {v: k for k, v in self.tag_to_id.iteritems()}
        self.id_to_word = {v: k for k, v in self.word_to_id.iteritems()}
        self.id_to_char = {v: k for k, v in self.char_to_id.iteritems()}

        self.ner_vocab_size = len(self.id_to_tag)
        self.word_vocab_size = len(self.id_to_word)
        self.char_vocab_size = len(self.id_to_char)

        print self.word_to_id
        print self.id_to_word
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
            word_set.add(word)

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
            for line in fin:
                if line.strip() == "":
                    self.read_one_line_set(to_read_line, tag_set, word_set, char_set)
                    to_read_line = []
                else:
                    to_read_line.append(line.strip())
            self.read_one_line_set(to_read_line, tag_set, word_set, char_set)
        # tag_set = set(tag_list)
        # word_set = set(tag_list)
        # char_set = set(char_list)

        tag_vocab = self.get_vocab(tag_set)
        word_vocab = self.get_vocab(word_set, 1)
        char_vocab = self.get_vocab(char_set, 1)

        return tag_vocab, word_vocab, char_vocab

    def get_data_set(self, path, lang):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []

        def add_sent(one_sent):
            temp_sent = []
            temp_ner = []
            temp_char = []

            for w in one_sent:
                fields = w.split()
                word = fields[0]
                ner_tag = fields[-1]

                temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_ner.append(self.tag_to_id[ner_tag])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])
            sents.append(temp_sent)
            char_sents.append(temp_char)
            tgt_tags.append(temp_ner)
            discrete_features.append(get_feature_w(lang, one_sent)[0] if self.use_discrete_feature else [])

        with codecs.open(path, "r", "utf-8") as fin:
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent)
                    one_sent = []
                else:
                    one_sent.append(line.strip())
            if len(one_sent) > 0:
                add_sent(one_sent)

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0
        return sents, char_sents, tgt_tags, discrete_features

    def get_lr_test(self, path, lang):
        sents = []
        char_sents = []
        discrete_features = []

        def add_sent(one_sent):
            temp_sent = []
            temp_char = []
            temp_discrete = []
            for word in one_sent:
                if self.use_discrete_feature:
                    temp_discrete.append(get_feature_w(lang, word))
                temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])
            sents.append(temp_sent)
            char_sents.append(temp_char)
            discrete_features.append(temp_discrete)

        original_sents = []
        with codecs.open(path, "r", "utf-8") as fin:
            lineNum =1
            for line in fin:
                one_sent = line.rstrip().split()
                if line:
                    add_sent(one_sent)
                    original_sents.append(one_sent)

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        return sents, char_sents, discrete_features, original_sents