__author__ = 'chuntingzhou'
import dynet as dy
import numpy as np
from collections import defaultdict
import gzip
import cPickle as pkl

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def get_pretrained_emb(path_to_emb):
    a = 1
    b = 2
    return a, b


def get_feature_w(w):
    return [1, 0]


def pkl_dump(obj, path):
    with open(path, "wb") as fout:
        pkl.dump(obj, fout)


def pkl_load(path):
    with open(path, "rb") as fin:
        obj = pkl.load(fin)
    return obj

def log_sum_exp_dim_0(x):
    # numerically stable log_sum_exp
    dims = x.dim()
    max_score = dy.max_dim(x, 0) # (dim_1, batch_size)
    if len(dims[0]) == 1:
        max_score_extend = max_score
    else:
        max_score_reshape = dy.reshape(max_score, (1, dims[0][1]), batch_size=dims[1])
        max_score_extend = dy.concatenate([max_score_reshape] * dims[0][0])
    x = x - max_score_extend
    exp_x = dy.exp(x)
    # (dim_1, batch_size), if no dim_1, return ((1,), batch_size)
    log_sum_exp_x = dy.log(dy.mean_dim(exp_x, d=0) * dims[0][0])
    return log_sum_exp_x + max_score


def data_iterator(data_pair, batch_size):
    batches = make_bucket_batches(data_pair, batch_size)
    for batch in batches:
        yield batch


def make_bucket_batches(data_collections, batch_size):
    # Data are bucketed according to the length of the first item in the data_collections.
    buckets = defaultdict(list)
    tot_items = len(data_collections[0])
    for data_item in data_collections:
        src = data_item[0]
        buckets[len(src)].append(data_item)

    batches = []
    np.random.seed(2)
    for src_len in buckets:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)
        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batches.append([[bucket[i * batch_size + j][k] for j in range(cur_batch_size)] for k in range(tot_items)])
    np.random.shuffle(batches)
    return batches


def transpose_input(seq, padding_token):
    # input seq: list of samples [[w1, w2, ..], [w1, w2, ..]]
    max_len = max([len(sent) for sent in seq])
    seq_pad = []
    seq_mask = []
    for i in range(max_len):
        pad_temp = [sent[i] if i < len(sent) else padding_token for sent in seq]
        mask_temp = [1.0 if i < len(sent) else 0.0 for sent in seq]
        seq_pad.append(pad_temp)
        seq_mask.append(mask_temp)

    return seq_pad, seq_mask


def transpose_discrete_features(feature_batch):
    # Discrete features are zero-one features
    # TODO: Other integer features, create lookup tables
    # tgt_batch: [[[feature of word 1 of sent 1], [feature of word 2 of sent 2], ]]
    # return: [(feature_num, batchsize)]
    max_sent_len = max([len(s) for s in feature_batch])
    feature_num = len(feature_batch[0][0])
    batch_size = len(feature_batch)
    features = [] # each: (feature_num, batch_size)
    for i in range(max_sent_len):
        w_i_feature = [dy.inputTensor(sent[i], batched=True) if i < len(sent) else dy.zeros(feature_num) for sent in feature_batch]
        w_i_feature = dy.reshape(dy.concatenate(w_i_feature, d=1), (feature_num,), batch_size=batch_size)
        features.append(w_i_feature)

    return features


def transpose_and_batch_embs(input_embs, emb_size):
    # input_embs: [[w1_emb, w2_emb, ]], embs are dy.expressions
    max_len = max(len(sent) for sent in input_embs)
    batch_size = len(input_embs)
    padded_seq_emb = []
    seq_masks = []
    for i in range(max_len):
        w_i_emb = [sent[i] if i < len(sent) else dy.zeros(emb_size) for sent in input_embs]
        w_i_emb = dy.reshape(dy.concatenate(w_i_emb, d=1), (emb_size, ), batch_size=batch_size)
        w_i_mask = [1.0 if i < len(sent) else 0.0 for sent in input_embs]
        padded_seq_emb.append(w_i_emb)
        seq_masks.append(w_i_mask)

    return padded_seq_emb, seq_masks


def transpose_char_input(tgt_batch, padding_token):
    # The tgt_batch may not be padded with <sow> and <eow>
    # tgt_batch: [[[<sow>, <sos>, <eow>], [<sow>, s,h,e, <eow>],
    # [<sow>, i,s, <eow>], [<sow>, p,r,e,t,t,y, <eow>], [<sow>, <eos>, <eow>]], [[],[],[]]]
    max_sent_len = max([len(s) for s in tgt_batch])
    sent_w_batch = []  # each is list of list: max_word_len, batch_size
    sent_mask_batch = []  # each is list of list: max_word_len, batch_size
    max_w_lens = []
    SOW_PAD = 0
    EOW_PAD = 1
    EOS_PAD = 2
    for i in range(max_sent_len):
        max_len_w = max([len(sent[i]) for sent in tgt_batch if i < len(sent)])
        max_w_lens.append(max_len_w)
        w_batch = []
        mask_batch = []
        for j in range(0, max_len_w):
            temp_j_w = []
            for sent in tgt_batch:
                if i < len(sent) and j < len(sent[i]):
                    temp_j_w.append(sent[i][j])
                elif i >= len(sent):
                    if j == 0:
                        temp_j_w.append(SOW_PAD)
                    elif j == max_len_w - 1:
                        temp_j_w.append(EOW_PAD)
                    else:
                        temp_j_w.append(EOS_PAD)
                else:
                    temp_j_w.append(EOW_PAD)
            # w_batch = [sent[i][j] if i < len(sent) and j < len(sent[i]) else self.EOW for sent in tgt_batch]
            # print "temp: ", temp_j_w
            w_batch.append(temp_j_w)
            mask_batch.append([1. if i < len(sent) and j < len(sent[i]) else 0.0 for sent in tgt_batch])
        sent_w_batch.append(w_batch)
        sent_mask_batch.append(mask_batch)
    return sent_w_batch, sent_mask_batch, max_sent_len, max_w_lens