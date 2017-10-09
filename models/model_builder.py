__author__ = 'chuntingzhou'
from encoders import *
from decoders import *


class Model(object):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError


class vanilla_NER_CRF_model(Model):
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
        birnn_input_dim = args.cnn_filter_size + args.word_emb_dim
        hidden_dim = args.hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size
        dropout_rate = args.dropout_rate

        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            birnn_input_dim += args.feature_dim

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size, dropout_rate, char_vocab_size)
        self.word_lookup = Lookup_Encoder(self.model, word_vocab_size, word_emb_dim, word_padding_token)

        self.birnn_encoder = BiRNN_Encoder(self.model, birnn_input_dim, hidden_dim, dropout_rate)

        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size)

    def forward(self, sents, char_sents, feats):
        char_embs = self.char_cnn_encoder.encode(char_sents)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]
        birnn_outputs = self.birnn_encoder.encode(concat_inputs)

        return birnn_outputs

    def cal_loss(self, sents, char_sents, ner_tags, feats):
        birnn_outputs = self.forward(sents, char_sents, feats)
        crf_loss = self.crf_decoder.decode_loss(birnn_outputs, ner_tags)
        return crf_loss

    def eval(self, sents, char_sents, feats):
        birnn_outputs = self.forward(sents, char_sents, feats)
        best_score, best_path = self.crf_decoder.decoding(birnn_outputs)
        return best_score, best_path