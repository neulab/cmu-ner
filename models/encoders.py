__author__ = 'chuntingzhou'
from utils import *

''' Designing idea: the encoder should be agnostic to the input, it can be either
    arbitrary spans, characters, or words, or even raw feature. However, user has to specify
    whether to have the lookup table for any input.

    There are also two ways to feed in multiple input features:
    (a) First concatenate all features for each position, and then use them as features for one encoder, e.g. bilstm
    (b) Use multiple encoders for multiple features then combine outputs from multiple encoders, either concat them
        or feed them to another encoder.'''


class Encoder():
    def __init__(self):
        pass

    def encode(self):
        raise NotImplementedError

# class concat_input_encoder(encoder):
#     def __init__(self, model, lookups, lookup_table_dims):
#         # length of elements in lookup_table_dims == number of elements in lookups which are true
#         self.num_inputs = len(lookups)
#         self.lookups = lookups
#         self.lookup_params = []
#         for i, lookup in enumerate(lookups):
#             if lookup == 1:
#                 # add loop up parameters
#                 self.lookup_params.append(model.add_lookup_parameters((lookup_table_dims[i][0], lookup_table_dims[i][1])))
#             elif lookup == 2:
#                 # add normal transformation parameters
#                 # dims: discrete_feature_num, continuous_emb_dim
#                 # the input should concatenate all the discrete features together first
#                 self.lookup_params.append(model.add_parameters((lookup_table_dims[i][0], lookup_table_dims[i][1])))
#             else:
#                 self.lookup_params.append(0)
#
#     def prepare_inputs(self, inputs):
#         # inputs: (a)
#         input_features = []
#         for i, lookup in enumerate(self.lookups):
#             if lookup == 1:


class Lookup_Encoder(Encoder):
    def __init__(self, model, vocab_size, emb_size, padding_token=None, pretrain_embedding=None):
        Encoder.__init__(self)
        self.padding_token = padding_token
        if pretrain_embedding is not None:
            self.lookup_table = model.add_lookup_parameters((vocab_size, emb_size), init=dy.NumpyInitializer(pretrain_embedding))
        else:
            self.lookup_table = model.add_lookup_parameters((vocab_size, emb_size))

    def encode(self, input_seqs):
        transpose_inputs, _ = transpose_input(input_seqs, self.padding_token)
        embs = [dy.lookup_batch(self.lookup_table, wids) for wids in transpose_inputs]
        return embs


class Discrete_Feature_Encoder(Encoder):
    def __init__(self, model, num_feats, to_dim):
        Encoder.__init__(self)
        self.num_feats = num_feats
        self.to_dim = to_dim
        self.W_feat_emb = model.add_parameters((to_dim, num_feats))

    def encode(self, input_feats):
        # after transpose: input_feats: [(num_feats, batch_size)]
        input_feats = transpose_discrete_features(input_feats)
        W_feat_emb = dy.parameter(self.W_feat_emb)
        output_emb = [W_feat_emb * ii for ii in input_feats]
        return output_emb


class CNN_Encoder(Encoder):
    def __init__(self, model, emb_size, win_size=3, filter_size=64, dropout=0.5, vocab_size=0, padding_token=0):
        Encoder.__init__(self)
        self.vocab_size = vocab_size # if 0, no lookup tables
        self.win_size = win_size
        self.filter_size =filter_size
        self.emb_size = emb_size
        self.dropout_rate=dropout
        self.paddding_token = padding_token
        if vocab_size != 0:
            print "In CNN encoder: creating lookup embedding!"
            self.lookup_emb = model.add_lookup_parameters((vocab_size, 1, 1, emb_size))
        self.W_cnn = model.add_parameters((1, win_size, emb_size, filter_size))
        self.b_cnn = model.add_parameters((filter_size))

    def _cnn_emb(self, input_embs):
        # input_embs: (h, time_step, dim, batch_size), h=1
        if self.dropout_rate > 0:
            input_embs = dy.dropout(input_embs, self.dropout_rate)
        W_cnn = dy.parameter(self.W_cnn)
        b_cnn = dy.parameter(self.b_cnn)

        cnn_encs = dy.conv2d_bias(input_embs, W_cnn, b_cnn, stride=(1, 1), is_valid=False)
        max_pool_out = dy.reshape(dy.max_dim(cnn_encs, d=1), (self.filter_size,))
        rec_pool_out = dy.rectify(max_pool_out)
        return rec_pool_out

    def encode(self, input_seqs, char=True):
        batch_size = len(input_seqs)
        sents_embs = []
        if char:
            # we don't batch at first, we batch after cnn
            for sent in input_seqs:
                sent_emb = []
                for w in sent:
                    if len(w) < self.win_size:
                        w += [self.paddding_token] * (self.win_size - len(w))
                    input_embs = dy.concatenate([dy.lookup(self.lookup_emb, c) for c in w], d=1)
                    w_emb = self._cnn_emb(input_embs)  # (filter_size, 1)
                    sent_emb.append(w_emb)
                sents_embs.append(sent_emb)
            sents_embs, sents_mask = transpose_and_batch_embs(sents_embs, self.filter_size) # [(filter_size, batch_size)]
        else:
            for sent in input_seqs:
                if self.vocab_size != 0:
                    if len(sent) < self.win_size:
                        sent += [0] * (self.win_size - len(sent))
                    input_embs = dy.concatenate([dy.lookup(self.lookup_emb, w) for w in sent], d=1)
                else:
                    # input_seqs: [(emb_size, batch_size)]
                    if len(sent) < self.win_size:
                        sent += [dy.zeros(self.emb_size)] * (self.win_size - len(sent))
                    input_embs = dy.transpose(dy.concatenate_cols(sent)) # (time_step, emb_size, bs)
                    input_embs = dy.reshape(input_embs, (1, len(sent), self.emb_size), )

                sent_emb = self._cnn_emb(input_embs)  # (filter_size, 1)
                sents_embs.append(sent_emb)
            sents_embs = dy.reshape(dy.concatenate(sents_embs, d=1), (self.filter_size,), batch_size =batch_size) # (filter_size, batch_size)

        return sents_embs


class BiRNN_Encoder(Encoder):
    def __init__(self,
                 model,
                 input_dim,
                 hidden_dim,
                 dropout_rate=0.5,
                 padding_token=None,
                 vocab_size=0,
                 emb_size=0,
                 layer=1,
                 rnn="lstm"):
        Encoder.__init__(self)
        # self.birnn = dy.BiRNNBuilder(layer, input_dim, hidden_dim, model, dy.LSTMBuilder if rnn == "lstm" else dy.GRUBuilder)
        self.fwd_RNN = dy.VanillaLSTMBuilder(layer, input_dim, hidden_dim, model) if rnn == "lstm" else dy.GRUBuilder(layer, input_dim, hidden_dim, model)
        self.bwd_RNN = dy.VanillaLSTMBuilder(layer, input_dim, hidden_dim, model) if rnn == "lstm" else dy.GRUBuilder(layer, input_dim, hidden_dim, model)

        self.vocab_size =  vocab_size
        self.padding_token = padding_token
        self.drop_out_rate = dropout_rate
        if vocab_size > 0:
            print "In BiRNN, creating lookup table!"
            self.vocab_emb = model.add_lookup_parameters((vocab_size, emb_size))

    def encode(self, input_seqs):
        if self.vocab_size > 0:
            # input_seqs = [[w1, w2],[]]
            transpose_inputs, _ = transpose_input(input_seqs, self.padding_token)
            w_embs = [dy.dropout(dy.lookup_batch(self.vocab_emb, wids), self.drop_out_rate) if self.drop_out_rate > 0. else dy.lookup_batch(self.vocab_emb, wids) for wids in transpose_inputs]
        else:
            w_embs = [dy.dropout(emb, self.drop_out_rate) if self.drop_out_rate > 0. else emb for emb in input_seqs]
        # if vocab_size = 0: input_seqs = [(input_dim, batch_size)]

        w_embs_r = w_embs[::-1]
        # birnn_outputs = [dy.dropout(emb, self.drop_out_rate) if self.drop_out_rate > 0. else emb for emb in self.birnn.transduce(w_embs)]
        fwd_vectors = self.fwd_RNN.initial_state().transduce(w_embs)
        bwd_vectors = self.bwd_RNN.initial_state().transduce(w_embs_r)[::-1]

        birnn_outputs = [dy.concatenate([fwd_v, bwd_v]) for (fwd_v, bwd_v) in zip(fwd_vectors, bwd_vectors)]
        return birnn_outputs