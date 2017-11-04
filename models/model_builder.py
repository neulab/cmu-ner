__author__ = 'chuntingzhou'
from encoders import *
from decoders import *

np.set_printoptions(threshold='nan')


class CRF_Model(object):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError


class vanilla_NER_CRF_model(CRF_Model):
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        if args.map_pretrain:
            birnn_input_dim = args.char_hidden_dim * 2 + args.map_dim
        else:
            birnn_input_dim = args.cnn_filter_size + args.word_emb_dim
        hidden_dim = args.hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size
        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate

        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            birnn_input_dim += args.feature_dim

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)
        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)
            # print data_loader.word_to_id
            # for i in range(len(data_loader.word_to_id)):
            #     print i, data_loader.id_to_word[i]
            # print data_loader.pretrain_word_emb
            # print "*************************************"
            # for i in range(len(data_loader.word_to_id)):
            #     print self.word_lookup.lookup_table[i].npvalue()
            # raw_input()
        self.birnn_encoder = BiRNN_Encoder(self.model, birnn_input_dim, hidden_dim, emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size)

    def forward(self, sents, char_sents, feats, training=True):
        char_embs = self.char_cnn_encoder.encode(char_sents, training=training)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]
        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs

    def cal_loss(self, sents, char_sents, ner_tags, feats, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, training=training)
        crf_loss = self.crf_decoder.decode_loss(birnn_outputs, ner_tags)
        return crf_loss#, sum_s, sent_s

    def eval(self, sents, char_sents, feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, training=training)
        best_score, best_path = self.crf_decoder.decoding(birnn_outputs)
        return best_score, best_path


class BiRNN_CRF_model(CRF_Model):
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        if args.map_pretrain:
            birnn_input_dim = args.char_hidden_dim * 2 + args.map_dim
        else:
            birnn_input_dim = args.char_hidden_dim * 2 + args.word_emb_dim
        hidden_dim = args.hidden_dim
        char_hidden_dim = args.char_hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate
        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            birnn_input_dim += args.feature_dim

        self.char_birnn_encoder = BiRNN_Encoder(self.model,
                 char_emb_dim,
                 char_hidden_dim,
                 emb_dropout_rate=0.0,
                 output_dropout_rate=0.0,
                 vocab_size=char_vocab_size,
                 emb_size=char_emb_dim)

        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size)

    def forward(self, sents, char_sents, feats, training=True):
        char_embs = self.char_birnn_encoder.encode(char_sents, training=training, char=True)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]
        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs

    def cal_loss(self, sents, char_sents, ner_tags, feats, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, training=training)
        crf_loss = self.crf_decoder.decode_loss(birnn_outputs, ner_tags)
        return crf_loss#, sum_s, sent_s

    def eval(self, sents, char_sents, feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, training=training)
        best_score, best_path = self.crf_decoder.decoding(birnn_outputs)
        return best_score, best_path
