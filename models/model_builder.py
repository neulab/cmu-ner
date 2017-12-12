__author__ = 'chuntingzhou'
from encoders import *
from decoders import *

np.set_printoptions(threshold='nan')


class CRF_Model(object):
    def __init__(self, args, data_loader):
        self.save_to = args.save_to_path
        self.load_from = args.load_from_path
        tag_to_id = data_loader.tag_to_id
        self.constraints = [[[tag_to_id["B-GPE"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-ORG"]] * 3, [tag_to_id["I-GPE"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-PER"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-GPE"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["B-LOC"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-GPE"]]],
                            [[tag_to_id["O"]] * 4, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"], tag_to_id["I-GPE"]]],
                            [[tag_to_id["I-GPE"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-ORG"]] * 3, [tag_to_id["I-GPE"], tag_to_id["I-PER"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-PER"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-GPE"], tag_to_id["I-LOC"]]],
                            [[tag_to_id["I-LOC"]] * 3, [tag_to_id["I-ORG"], tag_to_id["I-PER"], tag_to_id["I-GPE"]]]]

        # print self.constraints

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        raise NotImplementedError

    def save(self):
        if self.save_to is not None:
            self.model.save(self.save_to)
        else:
            print('Save to path not provided!')

    def load(self, path=None):
        if path is None:
            path = self.load_from
        if self.load_from is not None or path is not None:
            print('Load model parameters from %s!' % path)
            self.model.populate(path)
        else:
            print('Load from path not provided!')

    def cal_loss(self, sents, char_sents, ner_tags, feats, bc_feats, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        crf_loss = self.crf_decoder.decode_loss(birnn_outputs, ner_tags)
        return crf_loss#, sum_s, sent_s

    def eval(self, sents, char_sents, feats, bc_feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        best_score, best_path = self.crf_decoder.decoding(birnn_outputs)
        return  best_score, best_path

    def eval_scores(self, sents, char_sents, feats, bc_feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        tag_scores, transit_score = self.crf_decoder.get_crf_scores(birnn_outputs)
        return tag_scores, transit_score


class vanilla_NER_CRF_model(CRF_Model):
    ''' Implement End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. '''
    def __init__(self, args, data_loader):
        super(vanilla_NER_CRF_model, self).__init__(args, data_loader)
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
            birnn_input_dim = args.cnn_filter_size + args.map_dim
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

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)
            birnn_input_dim += bc_dim

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)
        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "In NER CRF: Using pretrained word embedding!"
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
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs = self.char_cnn_encoder.encode(char_sents, training=training)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)

        if self.args.use_brown_cluster:
            bc_feat_embs = self.bc_encoder.encode(bc_feats)

        if self.args.use_discrete_features and self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f, b]) for c, w, f, b in
                             zip(char_embs, word_embs, feat_embs, bc_feat_embs)]
        elif self.args.use_brown_cluster and not self.args.use_discrete_features:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, bc_feat_embs)]
        elif self.args.use_discrete_features and not self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]

        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs


class BiRNN_CRF_model(CRF_Model):
    ''' The same as above, except that we replace the cnn layer for characters with BiRNN layer. '''
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        super(BiRNN_CRF_model, self).__init__(args, data_loader)
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

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)
            birnn_input_dim += bc_dim

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
            print "In NER CRF: Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs = self.char_birnn_encoder.encode(char_sents, training=training, char=True)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)

        if self.args.use_brown_cluster:
            bc_feat_embs = self.bc_encoder.encode(bc_feats)

        if self.args.use_discrete_features and self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f, b]) for c, w, f, b in
                             zip(char_embs, word_embs, feat_embs, bc_feat_embs)]
        elif self.args.use_brown_cluster and not self.args.use_discrete_features:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, bc_feat_embs)]
        elif self.args.use_discrete_features and not self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                             zip(char_embs, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]

        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs


class CNN_BiRNN_CRF_model(CRF_Model):
    ''' Concatenate both the cnn char representation and birnn char representation as the char vector. '''
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        super(CNN_BiRNN_CRF_model, self).__init__(args, data_loader)
        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        if args.map_pretrain:
            birnn_input_dim = args.char_hidden_dim * 2 + args.map_dim + args.cnn_filter_size
        else:
            birnn_input_dim = args.char_hidden_dim * 2 + args.word_emb_dim + args.cnn_filter_size
        hidden_dim = args.hidden_dim
        char_hidden_dim = args.char_hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate
        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            birnn_input_dim += args.feature_dim

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)
            birnn_input_dim += bc_dim

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)

        self.char_birnn_encoder = BiRNN_Encoder(self.model,
                                                char_emb_dim,
                                                char_hidden_dim,
                                                emb_dropout_rate=0.0,
                                                output_dropout_rate=0.0,
                                                vocab_size=0,
                                                emb_size=char_emb_dim,
                                                vocab_emb=self.char_cnn_encoder.lookup_emb)

        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "In NER CRF: Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate,
                                           vocab_size=0)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs_birnn = self.char_birnn_encoder.encode(char_sents, training=training, char=True)
        char_embs_cnn = self.char_cnn_encoder.encode(char_sents, training=training, char=True)
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_discrete_features:
            feat_embs = self.feature_encoder.encode(feats)

        if self.args.use_brown_cluster:
            bc_feat_embs = self.bc_encoder.encode(bc_feats)

        if self.args.use_discrete_features and self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([cr, cc, w, f, b]) for cr, cc, w, f, b in
                             zip(char_embs_birnn, char_embs_cnn, word_embs, feat_embs, bc_feat_embs)]
        elif self.args.use_brown_cluster and not self.args.use_discrete_features:
            concat_inputs = [dy.concatenate([cr, cc, w, f]) for cr, cc, w, f in
                             zip(char_embs_birnn, char_embs_cnn, word_embs, bc_feat_embs)]
        elif self.args.use_discrete_features and not self.args.use_brown_cluster:
            concat_inputs = [dy.concatenate([cr, cc, w, f]) for cr, cc, w, f in
                             zip(char_embs_birnn, char_embs_cnn, word_embs, feat_embs)]
        else:
            concat_inputs = [dy.concatenate([cr, cc, w]) for cr, cc, w in zip(char_embs_birnn, char_embs_cnn, word_embs)]

        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs


class Sep_Encoder_CRF_model(CRF_Model):
    ''' Difference with CNN_BiRnn_CRF_Model: use two BiLSTM to model the embedding features (char and word) and linguistic features respectively. '''
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        super(Sep_Encoder_CRF_model, self).__init__(args, data_loader)
        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        if args.map_pretrain:
            birnn_input_dim = args.char_hidden_dim * 2 + args.map_dim + args.cnn_filter_size
        else:
            birnn_input_dim = args.char_hidden_dim * 2 + args.word_emb_dim + args.cnn_filter_size
        hidden_dim = args.hidden_dim
        char_hidden_dim = args.char_hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate

        self.feature_birnn_input_dim = 0

        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            self.feature_birnn_input_dim += args.feature_dim

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)
            self.feature_birnn_input_dim += bc_dim

        if self.feature_birnn_input_dim > 0:
            self.feature_birnn = BiRNN_Encoder(self.model,
                                               self.feature_birnn_input_dim,
                                               args.feature_birnn_hidden_dim,
                                               emb_dropout_rate=0.0,
                                               output_dropout_rate=output_dropout_rate,
                                               vocab_size=0)
            src_ctx_dim += args.feature_birnn_hidden_dim * 2

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)

        self.char_birnn_encoder = BiRNN_Encoder(self.model,
                                                char_emb_dim,
                                                char_hidden_dim,
                                                emb_dropout_rate=0.0,
                                                output_dropout_rate=0.0,
                                                vocab_size=0,
                                                emb_size=char_emb_dim,
                                                vocab_emb=self.char_cnn_encoder.lookup_emb)

        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "In NER CRF: Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate,
                                           vocab_size=0)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs_birnn = self.char_birnn_encoder.encode(char_sents, training=training, char=True)
        char_embs_cnn = self.char_cnn_encoder.encode(char_sents, training=training, char=True)
        word_embs = self.word_lookup.encode(sents)

        concat_inputs = [dy.concatenate([cr, cc, w]) for cr, cc, w in zip(char_embs_birnn, char_embs_cnn, word_embs)]
        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        if self.feature_birnn_input_dim > 0:
            if self.args.use_discrete_features:
                feat_embs = self.feature_encoder.encode(feats)
                concat_inputs = feat_embs
            if self.args.use_brown_cluster:
                cluster_embs = self.bc_encoder.encode(bc_feats)
                concat_inputs = cluster_embs

            if self.args.use_discrete_features and self.args.use_brown_cluster:
                concat_inputs = [dy.concatenate([fe, ce]) for fe, ce in
                                 zip(feat_embs, cluster_embs)]

            fts_birnn_outputs = self.feature_birnn.encode(concat_inputs, training=training)
            birnn_outputs = [dy.concatenate([eb, fb]) for eb, fb in zip(birnn_outputs, fts_birnn_outputs)]

        return birnn_outputs


class Sep_CNN_Encoder_CRF_model(CRF_Model):
    ''' Difference with CNN_BiRnn_CRF_Model: use two BiLSTM to model the embedding features (char and word) and linguistic features respectively. '''
    def __init__(self, args, data_loader):
        self.model = dy.Model()
        self.args = args
        super(Sep_CNN_Encoder_CRF_model, self).__init__(args, data_loader)
        ner_tag_size = data_loader.ner_vocab_size
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim
        if args.map_pretrain:
            birnn_input_dim = args.map_dim + args.cnn_filter_size
        else:
            birnn_input_dim = args.word_emb_dim + args.cnn_filter_size
        hidden_dim = args.hidden_dim
        src_ctx_dim = args.hidden_dim * 2

        cnn_filter_size = args.cnn_filter_size
        cnn_win_size = args.cnn_win_size

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate

        self.feature_birnn_input_dim = 0

        if args.use_discrete_features:
            self.num_feats = data_loader.num_feats
            self.feature_encoder = Discrete_Feature_Encoder(self.model, self.num_feats, args.feature_dim)
            self.feature_birnn_input_dim += args.feature_dim

        if args.use_brown_cluster:
            bc_num = args.brown_cluster_num
            bc_dim = args.brown_cluster_dim
            # for each batch, the length of input seqs are the same, so we don't have bother with padding
            self.bc_encoder = Lookup_Encoder(self.model, args, bc_num, bc_dim, word_padding_token, isFeatureEmb=True)
            self.feature_birnn_input_dim += bc_dim

        if self.feature_birnn_input_dim > 0:
            self.feature_birnn = BiRNN_Encoder(self.model,
                                               self.feature_birnn_input_dim,
                                               args.feature_birnn_hidden_dim,
                                               emb_dropout_rate=0.0,
                                               output_dropout_rate=output_dropout_rate,
                                               vocab_size=0)
            src_ctx_dim += args.feature_birnn_hidden_dim * 2

        self.char_cnn_encoder = CNN_Encoder(self.model, char_emb_dim, cnn_win_size, cnn_filter_size,
                                            0.0, char_vocab_size, data_loader.char_padding_token)


        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print "In NER CRF: Using pretrained word embedding!"
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate,
                                           vocab_size=0)

        # self.crf_decoder = classifier(self.model, src_ctx_dim, ner_tag_size)
        self.crf_decoder = chain_CRF_decoder(self.model, src_ctx_dim, tag_emb_dim, ner_tag_size, constraints=self.constraints)

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        char_embs_cnn = self.char_cnn_encoder.encode(char_sents, training=training, char=True)
        word_embs = self.word_lookup.encode(sents)

        concat_inputs = [dy.concatenate([cc, w]) for cc, w in zip(char_embs_cnn, word_embs)]
        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        if self.feature_birnn_input_dim > 0:
            if self.args.use_discrete_features:
                feat_embs = self.feature_encoder.encode(feats)
                concat_inputs = feat_embs
            if self.args.use_brown_cluster:
                cluster_embs = self.bc_encoder.encode(bc_feats)
                concat_inputs = cluster_embs

            if self.args.use_discrete_features and self.args.use_brown_cluster:
                concat_inputs = [dy.concatenate([fe, ce]) for fe, ce in
                                 zip(feat_embs, cluster_embs)]

            fts_birnn_outputs = self.feature_birnn.encode(concat_inputs, training=training)
            birnn_outputs = [dy.concatenate([eb, fb]) for eb, fb in zip(birnn_outputs, fts_birnn_outputs)]

        return birnn_outputs

