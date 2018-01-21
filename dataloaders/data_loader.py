__author__ = 'chuntingzhou'
import os
from utils.util import *

# from utils.segnerfts import orm_morph as ormnorm

class NER_DataLoader():
    def __init__(self, args, special_normal=False):
        # This is data loader as well as feature extractor!!
        '''Data format: id word pos_tag syntactic_tag NER_tag'''
        ''' TODO: 1. normalizing all digits
                  2. Using full vocabulary from GloVe, when testing, lower case first'''
        self.args = args
        if args.train_ensemble:
            self.train_path = args.full_data_path
        else:
            self.train_path = args.train_path
        self.test_path = args.test_path
        self.dev_path = args.dev_path
        self.args = args

        self.tag_vocab_path = self.train_path + ".tag_vocab"
        self.word_vocab_path = self.train_path + ".word_vocab"
        self.char_vocab_path = self.train_path + ".char_vocab"

        self.pretrained_embedding_path = args.pretrain_emb_path
        self.use_discrete_feature = args.use_discrete_features
        self.use_brown_cluster = args.use_brown_cluster
        self.orm_norm = args.oromo_normalize
        self.orm_lower = args.train_lowercase_oromo

        if special_normal:
            self.orm_norm = False
            self.orm_lower = False

        if self.use_brown_cluster:
            self.brown_cluster_dicts = get_brown_cluster(args.brown_cluster_path)
            self.brown_cluster_dicts['<unk>'] = 499
        else:
            self.brown_cluster_dicts = None

        if False and os.path.exists(self.tag_vocab_path) and os.path.exists(self.word_vocab_path) and os.path.exists(self.char_vocab_path):
            # TODO: encoding?
            print("Load vocabs from file ....")
            self.tag_to_id = pkl_load(self.tag_vocab_path)
            self.word_to_id = pkl_load(self.word_vocab_path)
            self.char_to_id = pkl_load(self.char_vocab_path)
            print("Done!")
        else:
            print("Generating vocabs from training file ....")
            if not self.args.isLr:
                paths_to_read = [self.train_path, self.test_path, self.dev_path]
                self.tag_to_id, self.word_to_id, self.char_to_id = self.read_files(paths_to_read)
            else:
                paths_to_read = [self.train_path]
                setEpaths = [self.dev_path, self.test_path]
                self.tag_to_id, self.word_to_id, self.char_to_id = self.read_files_lr(paths_to_read, setEpaths)
            # FIXME: Remember dictionary value for char and word has been shifted by 1
            print "Size of vocab before: ", len(self.word_to_id)
            self.word_to_id['<unk>'] = len(self.word_to_id) + 1
            self.char_to_id['<unk>'] = len(self.char_to_id) + 1

            self.word_to_id['<\s>'] = 0
            self.char_to_id['<pad>'] = 0
            print "Size of vocab after: ", len(self.word_to_id)
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

        print "Size of vocab after: ", len(self.word_to_id)
        print("NER tag num=%d, Word vocab size=%d, Char Vocab size=%d" % (self.ner_vocab_size, self.word_vocab_size, self.char_vocab_size))

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def read_one_line(self, line, tag_set, word_dict, char_set):
        for w in line:
            fields = w.split()
            word = fields[0]
            ner_tag = fields[-1]
            for c in word:
                char_set.add(c)
            tag_set.add(ner_tag)
            if self.orm_lower:
                word = word.lower()
            if self.orm_norm:
                #word = orm_morph.best_parse(word)
                word = ormnorm.normalize(word)
            word_dict[word] += 1

    def get_vocab_from_set(self, a_set, shift=0):
        vocab = {}
        for i, elem in enumerate(a_set):
            vocab[elem] = i + shift

        return vocab

    def get_vocab_from_dict(self, a_dict, shift=0, remove_singleton=False):
        vocab = {}
        i = 0
        self.singleton_words = set()
        for k, v in a_dict.iteritems():
            if v == 1:
                self.singleton_words.add(i + shift)
            if remove_singleton:
                if v > 1:
                    # print k, v
                    vocab[k] = i + shift
                    i += 1
            else:
                vocab[k] = i + shift
                i += 1
        print "Singleton words number: ", len(self.singleton_words)
        return vocab

    def read_files(self, paths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_set = set()

        def _read_a_file(path):
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                for line in fin:
                    if line.strip() == "":
                        self.read_one_line(to_read_line, tag_set, word_dict, char_set)
                        to_read_line = []
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line(to_read_line, tag_set, word_dict, char_set)

        for path in paths:
            _read_a_file(path)

        tag_vocab = self.get_vocab_from_set(tag_set)
        word_vocab = self.get_vocab_from_dict(word_dict, 1, self.args.remove_singleton)
        char_vocab = self.get_vocab_from_set(char_set, 1)

        return tag_vocab, word_vocab, char_vocab

    def read_files_lr(self, paths, setEpaths):
        # word_list = []
        # char_list = []
        # tag_list = []
        word_dict = defaultdict(lambda: 0)
        char_set = set()
        tag_set = set()

        def _read_a_file(path):
            with codecs.open(path, "r", "utf-8") as fin:
                to_read_line = []
                for line in fin:
                    if line.strip() == "":
                        self.read_one_line(to_read_line, tag_set, word_dict, char_set)
                        to_read_line = []
                    else:
                        to_read_line.append(line.strip())
                self.read_one_line(to_read_line, tag_set, word_dict, char_set)

        for path in paths:
            _read_a_file(path)

        #reading from SetE
        for path in setEpaths:
            with codecs.open(path, "r", "utf-8") as fin:
                for line in fin:
                    fields = line.strip().split()
                    for word in fields:
                        for c in word:
                            char_set.add(c)
                        if self.orm_lower:
                            word = word.lower()
                        if self.orm_norm:
                            #word = orm_morph.best_parse(word)
                            word = ormnorm.normalize(word)
                        word_dict[word] += 1

        tag_vocab = self.get_vocab_from_set(tag_set)
        word_vocab = self.get_vocab_from_dict(word_dict, 1, self.args.remove_singleton)
        char_vocab = self.get_vocab_from_set(char_set, 1)

        return tag_vocab, word_vocab, char_vocab

    def get_data_set(self, path, lang):
        sents = []
        char_sents = []
        tgt_tags = []
        discrete_features = []
        bc_features = []

        def add_sent(one_sent):
            temp_sent = []
            temp_ner = []
            temp_char = []
            temp_bc = []
            for w in one_sent:
                fields = w.split()
                word = fields[0]
                ner_tag = fields[-1]
                if self.use_brown_cluster:
                    temp_bc.append(self.brown_cluster_dicts[word] if word in self.brown_cluster_dicts else self.brown_cluster_dicts["<unk>"])

                if self.orm_lower:
                    word = word.lower()

                if self.orm_norm:
                    #word = orm_morph.best_parse(word) # Not sure whether it would be better adding this line behind or after temp_char
                    word = ormnorm.normalize(word)
                temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_ner.append(self.tag_to_id[ner_tag])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            tgt_tags.append(temp_ner)
            bc_features.append(temp_bc)
            if not self.args.isLr:
                discrete_features.append([])
            else:
                discrete_features.append(get_feature_sent(lang, one_sent, self.args) if self.use_discrete_feature else [])

            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent)
                        i += 1
                        if i % 1000 == 0:
                            print("Processed %d training data." % (i,))
                    one_sent = []
                else:
                    one_sent.append(line.strip())

            if len(one_sent) > 0:
                add_sent(one_sent)

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0
        return sents, char_sents, tgt_tags, discrete_features, bc_features

    def get_lr_test(self, path, lang):
        # setE.txt
        sents = []
        char_sents = []
        discrete_features = []
        bc_features = []

        def add_sent(one_sent):
            temp_sent = []
            temp_char = []
            temp_bc = []
            for word in one_sent:
                if self.use_brown_cluster:
                    temp_bc.append(self.brown_cluster_dicts[word] if word in self.brown_cluster_dicts else self.brown_cluster_dicts["<unk>"])
                if self.orm_lower:
                    word = word.lower()
                if self.orm_norm:
                    #word = orm_morph.best_parse(word) # Not sure whether it would be better adding this line behind or after temp_char
                    word = ormnorm.normalize(word)
                temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            sents.append(temp_sent)
            char_sents.append(temp_char)
            discrete_features.append(get_feature_sent(lang, one_sent, self.args) if self.use_discrete_feature else [])
            bc_features.append(temp_bc)

        original_sents = []
        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            for line in fin:
                one_sent = line.rstrip().split()
                if line:
                    add_sent(one_sent)
                    original_sents.append(one_sent)
                i += 1
                if i % 1000 == 0:
                    print("Processed %d testing data." % (i,))

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        return sents, char_sents, discrete_features, original_sents, bc_features

    def get_lr_test_setE(self, path, lang):
        # setE.conll
        sents = []
        char_sents = []
        discrete_features = []
        bc_features = []
        doc_ids = []
        original_sents = []

        def add_sent(one_sent):
            temp_sent = []
            temp_char = []
            temp_bc = []
            temp_ori_sent = []
            for w in one_sent:
                tokens = w.split('\t')
                word = tokens[0]
                temp_ori_sent.append(word)
                docfile = tokens[3]
                doc_type = docfile.split('_')[1]
                if self.use_brown_cluster:
                    temp_bc.append(self.brown_cluster_dicts[word] if word in self.brown_cluster_dicts else self.brown_cluster_dicts["<unk>"])

                if self.orm_lower:
                    word = word.lower()

                if self.orm_norm:
                    #word = orm_morph.best_parse(word)
                    word = ormnorm.normalize(word)

                temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            doc_ids.append(docfile.split('_')[1])
            sents.append(temp_sent)
            char_sents.append(temp_char)
            bc_features.append(temp_bc)
            discrete_features.append(get_feature_sent(lang, one_sent, self.args) if self.use_discrete_feature else [])
            original_sents.append(temp_ori_sent)
            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent)
                    one_sent = []
                else:
                    one_sent.append(line.strip())
                i += 1
                if i % 1000 == 0:
                    print("Processed %d testing data." % (i,))

            if len(one_sent) > 0:
                add_sent(one_sent)

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        return sents, char_sents, discrete_features, bc_features, original_sents, doc_ids


class Dataloader_Combine():
    def __init__(self, args, normal_vocab, lower_vocab, char_to_id, brown_cluster_dicts=None, lower_brown_dicts=None):
        self.word_to_id = normal_vocab
        self.lower_word_to_id = lower_vocab
	self.args = args

        self.char_to_id = char_to_id
        self.brown_cluster_dicts = brown_cluster_dicts
        self.lower_brown_dicts = lower_brown_dicts

        self.use_discrete_feature = args.use_discrete_features
        self.use_brown_cluster = args.use_brown_cluster
        self.orm_norm = args.oromo_normalize
        self.orm_lower = args.train_lowercase_oromo

    def get_lr_test_setE(self, path, lang):
        # setE.conll
        sents = []
        char_sents = []
        discrete_features = []
        bc_features = []
        doc_ids = []
        original_sents = []

        def add_sent(one_sent):
            temp_sent = []
            temp_char = []
            temp_bc = []
            temp_ori_sent = []
            for w in one_sent:
                tokens = w.split('\t')
                word = tokens[0]
                temp_ori_sent.append(word)
                docfile = tokens[3]
                doc_type = docfile.split('_')[1]
                if self.use_brown_cluster:
                    if doc_type == "SN":
                        temp_bc.append(self.lower_brown_dicts[word] if word in self.lower_brown_dicts else self.lower_brown_dicts["<unk>"])
                    else:
                        temp_bc.append(self.brown_cluster_dicts[word] if word in self.brown_cluster_dicts else self.brown_cluster_dicts["<unk>"])

                if doc_type == "SN":
                    if self.orm_lower:
                        word = word.lower()

                    if self.orm_norm:
                        #word = orm_morph.best_parse(word) # Not sure whether it would be better adding this line behind or after temp_char
                        word = ormnorm.normalize(word)
                    temp_sent.append(self.lower_word_to_id[word] if word in self.lower_word_to_id else self.lower_word_to_id["<unk>"])
                else:
                    temp_sent.append(self.word_to_id[word] if word in self.word_to_id else self.word_to_id["<unk>"])
                temp_char.append([self.char_to_id[c] if c in self.char_to_id else self.char_to_id["<unk>"] for c in word])

            doc_ids.append(docfile.split('_')[1])
            sents.append(temp_sent)
            char_sents.append(temp_char)
            bc_features.append(temp_bc)
            discrete_features.append(get_feature_sent(lang, one_sent, self.args) if self.use_discrete_feature else [])
            original_sents.append(temp_ori_sent)
            # print len(discrete_features[-1])

        with codecs.open(path, "r", "utf-8") as fin:
            i = 0
            one_sent = []
            for line in fin:
                if line.strip() == "":
                    if len(one_sent) > 0:
                        add_sent(one_sent)
                    one_sent = []
                else:
                    one_sent.append(line.strip())
                i += 1
                if i % 1000 == 0:
                    print("Processed %d testing data." % (i,))

            if len(one_sent) > 0:
                add_sent(one_sent)

        if self.use_discrete_feature:
            self.num_feats = len(discrete_features[0][0])
        else:
            self.num_feats = 0

        return sents, char_sents, discrete_features, bc_features, original_sents, doc_ids
