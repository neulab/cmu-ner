__author__ = 'chuntingzhou'
from utils import *
import copy

from mst import mst


class Decoder():
    def __init__(self):
        # type: () -> object
        pass

    def decode_loss(self):
        raise NotImplementedError

    def decoding(self):
        raise NotImplementedError


class chain_CRF_decoder(Decoder):
    ''' For NER and POS Tagging. '''

    def __init__(self, model, src_output_dim, tag_emb_dim, tag_size):
        Decoder.__init__(self)
        self.model = model

        self.start_id = tag_size
        self.end_id = tag_size + 1
        self.tag_size = tag_size + 2
        # optional: transform the hidden space of src encodings into the tag embedding space
        self.W_src2tag_readout = model.add_parameters((tag_emb_dim, src_output_dim))
        self.b_src2tag_readout = model.add_parameters((tag_emb_dim))
        self.b_src2tag_readout.zero()

        self.W_scores_readout2tag = model.add_parameters((tag_size, tag_emb_dim))
        self.b_scores_readout2tag = model.add_parameters((tag_size))
        self.b_scores_readout2tag.zero()

        # (to, from), trans[i] is the transition score to i
        init_transition_matrix = np.random.randn((tag_size, tag_size))
        init_transition_matrix[self.start_id, :] = -10000.0
        init_transition_matrix[:, self.end_id] = -10000.0
        self.transition_matrix = model.add_lookup_parameters((tag_size, tag_size),
                                                             init=dy.NumpyInitializer(init_transition_matrix))

    def forward_alg(self, tag_scores, batch_size):
        ''' Forward DP for CRF.
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        transition_score = dy.parameter(self.transition_matrix)
        transpose_transition_score = dy.transpose(transition_score)  # (from, to)
        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        np_init_alphas = -10000.0 * np.ones((self.tag_size, batch_size))
        np_init_alphas[self.start_id, :] = 0.0
        alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)

        for tag_score in tag_scores:
            # extend for each transit <to>
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)

        terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + transition_score[self.end_id])  # (1, batch_size)
        return terminal_alpha

    def score_one_sequence(self, tag_scores, tags, batch_size):
        ''' tags: list of tag ids at each time step '''
        tags = [self.start_id] * batch_size + tags  # len(tag_scores) = len(tags) - 1
        score = dy.inputTensor(np.zeros(batch_size), batched=True)
        for i in range(len(tags) - 1):
            score = score + dy.pick_batch(dy.lookup_batch(self.transition_matrix, tags[i + 1]), tags[i]) \
                    + dy.pick_batch(tag_scores[i], tags[i + 1])
        score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, [self.end_id] * batch_size), tags[-1])
        return score

    def decode_loss(self, src_encodings, tgt_tags):
        # This is the batched version which requires bucketed batch input with the same length.
        '''
        The length of src_encodings and tgt_tags are time_steps.
        src_encodings: list of dynet.Tensor (src_output_dim, batch_size)
        tgt_tags: list of tag ids [(1, batch_size)]
        return: average of negative log likelihood
        '''
        # TODO: transpose tgt tags first
        batch_size = len(tgt_tags[0])
        tgt_tags, tgt_mask = transpose_input(tgt_tags, 0)
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding])) for src_encoding
                    in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        # scores over all paths, all scores are in log-space
        forward_scores = self.forward_alg(tag_scores, batch_size)
        gold_score = self.score_one_sequence(tag_scores, tgt_tags, batch_size)
        loss = dy.sum_batches(gold_score - forward_scores) / batch_size
        return loss

    def decoding(self, src_encodings):
        ''' Viterbi decoding for a single sequence. '''
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        back_trace_tags = []
        np_init_alpha = np.ones(self.tag_size) * -10000.0
        np_init_alpha[self.start_id] = 0.0
        max_tm1 = dy.inputTensor(np_init_alpha)
        transition_score = dy.parameter(self.transition_matrix)  # (to, from)
        transpose_transition_score = dy.transpose(transition_score)  # (from, to)

        for tag_score in tag_scores:
            max_tm1 = dy.concatenate_cols([max_tm1] * self.tag_size)
            max_t = max_tm1 + transpose_transition_score
            eval_score = max_t.npvalue()
            best_tag = np.argmax(eval_score, axis=0)
            back_trace_tags.append(best_tag)
            max_tm1 = dy.inputTensor(eval_score[best_tag, range(self.tag_size)]) + tag_score

        terminal_max_T = max_tm1 + self.transition_matrix[self.end_id]
        eval_terminal = terminal_max_T.npvalue()
        best_tag = np.argmax(eval_terminal, axis=0)
        best_path_score = eval_terminal[best_tag]

        best_path = [best_tag]
        for btpoint in reversed(back_trace_tags):
            best_tag = btpoint[best_tag]
            best_path.append(best_tag)
        start = best_path.pop()
        assert start == self.start_id
        best_path.reverse()
        return best_path_score, best_path

    def cal_accuracy(self, pred_path, true_path):
        return np.sum(np.equal(pred_path, true_path))/len(pred_path)


class RNN_decoder(Decoder):
    ''' For neural machine translation. '''

    def __init__(self, model, src_ctx_dim, input_dim, hidden_dim, attention_dim, tgt_vocab_size, tgt_emb_dim):
        Decoder.__init__(self)
        self.hidden_dim = hidden_dim
        self.tgt_voc_size = tgt_vocab_size
        self.tgt_word_emb = model.add_lookup_parameters((tgt_vocab_size, tgt_emb_dim))
        self.tgt_dec_rnn = dy.GRUBuilder(1, input_dim, hidden_dim, model)

        self.dec_rnn = dy.GRUBuilder(1, tgt_emb_dim + src_ctx_dim, hidden_dim, model)
        self.W_init_state = model.add_parameters((hidden_dim, src_ctx_dim))
        self.b_init_state = model.add_parameters(hidden_dim)
        self.b_init_state.zero()

        # use MLP attention
        self.W_att_hidden = model.add_parameters((attention_dim, hidden_dim))
        self.W_att_src = model.add_parameters((attention_dim, src_ctx_dim))
        self.V_att = model.add_parameters((attention_dim))

        # first concatenate the attentional vector with the hidden state of decoder
        self.W_readout = model.add_parameters((tgt_emb_dim, src_ctx_dim + hidden_dim))
        self.b_readout = model.add_parameters((tgt_emb_dim))
        self.b_readout.zero()

        self.W_softmax = model.add_parameters((tgt_vocab_size, tgt_emb_dim))
        self.b_softmax = model.add_parameters((tgt_vocab_size))
        self.b_softmax.zero()

        self.EOS = 1
        self.SOS = 2

    def transpose_input(self, seq):
        max_len = max([len(sent) for sent in seq])
        seq_pad = []
        seq_mask = []
        for i in range(max_len):
            pad_temp = [sent[i] if i < len(sent) else self.EOS for sent in seq]
            mask_temp = [1 if i < len(sent) else 0 for sent in seq]
            seq_pad.append(pad_temp)
            seq_mask.append(mask_temp)
        return seq_pad, seq_mask

    def attention(self, encoding, hidden, src_transform_encoding, src_len, batch_size):
        W_att_hid = dy.parameter(self.W_att_hidden)
        V_att = dy.parameter(self.V_att)

        att_mlp = dy.tanh(dy.colwise_add(src_transform_encoding, W_att_hid * hidden))

        att_weights = dy.reshape(V_att * att_mlp, (src_len,), batch_size)

        att_weights = dy.softmax(att_weights)  # (time_step, batch_size)
        att_ctx = encoding * att_weights  # src_ctx_dim, batch_size

        return att_ctx, att_weights

    def decode_loss(self, src_encodings, tgt_seq, dropout=0.0):
        ''' src_encodings: list of dynet.Tensor (src_ctx_dim, batch_size) '''
        src_len = len(src_encodings)
        # tgt_seqs are padded with <sos> and <eos> at the beginning and the end
        W_init_state = dy.parameter(self.W_init_state)
        b_init_state = dy.parameter(self.b_init_state)

        W_att_src = dy.parameter(self.W_att_src)
        W_readout = dy.parameter(self.W_readout)
        b_readout = dy.parameter(self.b_readout)

        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)

        # tgt sequence starts from <S>, ends at <\S>
        batch_size = len(tgt_seq)

        init_state = dy.tanh(dy.affine_transform([b_init_state, W_init_state, src_encodings[-1]]))
        dec_state = self.dec_rnn.initial_state([init_state])

        padded_tgt_seqs, tgt_masks = self.transpose_input(tgt_seq)
        max_len = max([len(sent) for sent in tgt_seq])
        att_ctx = dy.vecInput(self.hidden_dim * 2)

        src_encodings = dy.concatenate_cols(src_encodings)  # (src_ctx_dim, time_step, batch_size)
        src_transform_encodings = W_att_src * src_encodings
        losses = []
        for i in range(max_len - 1):
            input_t = dy.lookup_batch(self.tgt_word_emb, padded_tgt_seqs[i])
            dec_state = dec_state.add_input(dy.concatenate([input_t, att_ctx]))
            ht = dec_state.output()
            att_ctx, att_weights = self.attention(src_encodings, ht, src_transform_encodings, src_len, batch_size)

            read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([ht, att_ctx])]))

            if dropout > 0:
                read_out = dy.dropout(read_out, dropout)
            prediction = dy.affine_transform([b_softmax, W_softmax, read_out])

            loss = dy.pickneglogsoftmax_batch(prediction, padded_tgt_seqs[i + 1])

            if 0 in tgt_masks[i + 1]:
                mask_expr = dy.inputTensor(tgt_masks[i + 1], batched=True)
                loss = loss * mask_expr

            losses.append(loss)

        loss = dy.esum(losses)
        loss = dy.sum_batches(loss) / batch_size

        return loss

    def decoding(self, src_encodings, beam_size=5, max_len=50):
        ''' Beam search decoding. '''
        src_len = len(src_encodings)
        src_encodings = dy.concatenate_cols(src_encodings)
        W_init_state = dy.parameter(self.W_init_state)
        b_init_state = dy.parameter(self.b_init_state)

        W_att_src = dy.parameter(self.W_att_src)
        W_readout = dy.parameter(self.W_readout)
        b_readout = dy.parameter(self.b_readout)

        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)

        live = 1
        dead = 0

        final_scores = []
        final_samples = []

        scores = np.zeros(live)
        dec_states = [
            self.dec_rnn.initial_state([dy.tanh(dy.affine_transform([b_init_state, W_init_state, src_encodings[-1]]))])]
        att_ctxs = [dy.vecInput(self.hidden_dim * 2)]
        samples = [[self.SOS]]
        src_encodings = dy.concatenate_cols(src_encodings)  # (src_ctx_dim, time_step, 1)
        src_transform_encodings = W_att_src * src_encodings

        for ii in range(max_len):
            cand_scores = []
            for k in range(live):
                y_t = dy.lookup(self.tgt_word_emb, samples[k][-1])
                dec_states[k] = dec_states[k].add_input(dy.concatenate([y_t, att_ctxs[k]]))
                h_t = dec_states[k].output()
                att_ctx, att_weights = self.attention(src_encodings, h_t, src_transform_encodings, src_len, 1)
                att_ctxs[k] = att_ctx

                read_out = dy.tanh(dy.affine_transform([b_readout, W_readout, dy.concatenate([h_t, att_ctx])]))

                prediction = dy.log_softmax(W_softmax * read_out + b_softmax).npvalue()
                cand_scores.append(scores[k] - prediction)

            cand_scores = np.concatenate(cand_scores).flatten()
            ranks = cand_scores.argsort()[:(beam_size - dead)]

            cands_indices = ranks / self.tgt_word_emb
            cands_words = ranks % self.tgt_voc_size
            cands_scores = cand_scores[ranks]

            new_scores = []
            new_dec_states = []
            new_att_ctxs = []
            new_samples = []
            for idx, [bidx, widx] in enumerate(zip(cands_indices, cands_words)):
                new_scores.append(copy.copy(cands_scores[idx]))
                new_dec_states.append(dec_states[bidx])
                new_att_ctxs.append(att_ctxs[bidx])
                new_samples.append(samples[bidx] + [widx])

            scores = []
            dec_states = []
            att_ctxs = []
            samples = []

            for idx, sample in enumerate(new_samples):
                if new_samples[idx][-1] == self.EOS:
                    dead += 1
                    final_samples.append(new_samples[idx])
                    final_scores.append(new_scores[idx])
                else:
                    dec_states.append(new_dec_states[idx])
                    att_ctxs.append(new_att_ctxs[idx])
                    samples.append(new_samples[idx])
                    scores.append(new_scores[idx])
            live = beam_size - dead

            if dead == beam_size:
                break

        if live > 0:
            for idx in range(live):
                final_scores.append(scores[idx])
                final_samples.append(samples[idx])

        return final_scores, final_samples


class deep_BiAffineAttention_decoder(Decoder):
    ''' For MST dependency parsing (ICLR 2017)'''

    def __init__(self,
                 model,
                 n_labels,
                 src_ctx_dim=400,
                 n_arc_mlp_units=500,
                 n_label_mlp_units=100,
                 arc_mlp_dropout=0.33,
                 label_mlp_dropout=0.33):
        '''To reproduce the results of the original paper,
        requires (1) the encoder to be a 3-layer bilstm;
                 (2) dropout rate of bilstm to be 0.33
                 (3) pretrained embeddings and embedding dropout rate to be 0.33'''

        Decoder.__init__(self)
        self.src_ctx_dim = src_ctx_dim
        self.label_mlp_dropout = label_mlp_dropout
        self.arc_mlp_dropout = arc_mlp_dropout
        self.n_labels = n_labels
        self.W_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units, src_ctx_dim))
        self.b_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units))
        self.W_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units, src_ctx_dim))
        self.b_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units))

        self.W_label_hidden_to_head = model.add_parameters((n_label_mlp_units, src_ctx_dim))
        self.b_label_hidden_to_head = model.add_parameters((n_label_mlp_units))
        self.W_label_hidden_to_dep = model.add_parameters((n_label_mlp_units, src_ctx_dim))
        self.b_label_hidden_to_dep = model.add_parameters((n_label_mlp_units))

        self.U_arc_1 = model.add_parameters((n_arc_mlp_units, n_arc_mlp_units))
        self.u_arc_2 = model.add_parameters((n_arc_mlp_units))

        self.U_label_1 = [model.add_parameters((n_label_mlp_units, n_label_mlp_units)) for _ in range(n_labels)]
        self.u_label_2_2 = [model.add_parameters((n_label_mlp_units)) for _ in range(n_labels)]
        self.u_label_2_1 = [model.add_parameters((1, n_label_mlp_units)) for _ in range(n_labels)]
        self.b_label = [model.add_parameters((1)) for _ in range(n_labels)]

    def cal_scores(self, src_encodings):
        src_len = len(src_encodings)
        src_encodings = dy.concatenate_cols(src_encodings)  # src_ctx_dim, src_len, batch_size

        W_arc_hidden_to_head = dy.parameter(self.W_arc_hidden_to_head)
        b_arc_hidden_to_head = dy.parameter(self.b_arc_hidden_to_head)
        W_arc_hidden_to_dep = dy.parameter(self.W_arc_hidden_to_dep)
        b_arc_hidden_to_dep = dy.parameter(self.b_arc_hidden_to_dep)

        W_label_hidden_to_head = dy.parameter(self.W_label_hidden_to_head)
        b_label_hidden_to_head = dy.parameter(self.b_label_hidden_to_head)
        W_label_hidden_to_dep = dy.parameter(self.W_label_hidden_to_dep)
        b_label_hidden_to_dep = dy.parameter(self.b_label_hidden_to_dep)

        U_arc_1 = dy.parameter(self.U_arc_1)
        u_arc_2 = dy.parameter(self.u_arc_2)

        U_label_1 = [dy.parameter(x) for x in self.U_label_1]
        u_label_2_1 = [dy.parameter(x) for x in self.u_label_2_1]
        u_label_2_2 = [dy.parameter(x) for x in self.u_label_2_2]
        b_label = [dy.parameter(x) for x in self.b_label]

        h_arc_head = dy.rectify(dy.affine_transform(
            [b_arc_hidden_to_head, W_arc_hidden_to_head, src_encodings]))  # n_arc_ml_units, src_len, bs
        h_arc_dep = dy.rectify(dy.affine_transform([b_arc_hidden_to_dep, W_arc_hidden_to_dep, src_encodings]))
        h_label_head = dy.rectify(dy.affine_transform([b_label_hidden_to_head, W_label_hidden_to_head, src_encodings]))
        h_label_dep = dy.rectify(dy.affine_transform([b_label_hidden_to_dep, W_label_hidden_to_dep, src_encodings]))

        h_arc_head_transpose = dy.transpose(h_arc_head)
        h_label_head_transpose = dy.transpose(h_label_head)

        s_arc = h_arc_head_transpose \
                * dy.colwise_add(U_arc_1 * h_arc_dep, u_arc_2)  # src_len, src_len, bs (head, dep, bs)

        s_label = []
        for U_1, u_2_1, u_2_2, b in zip(U_label_1, u_label_2_1, u_label_2_2, b_label):
            e1 = h_label_head_transpose * U_1 * h_label_dep
            e2 = h_label_head_transpose * u_2_1 * dy.ones((1, src_len))
            e3 = dy.ones((src_len, 1)) * u_2_2 * h_label_dep
            s_label.append(e1 + e2 + e3 + b)
        return s_arc, s_label

    def decode_loss(self, src_encodings, tgt_seqs):
        """
        :param src_encodings: list of dy.Expressions [(src_ctx_dim, batch_size)]
        :param tgt_seqs: (tgt_heads, tgt_labels): list (length=batch_size) of (src_len)
        """
        # TODO: Sentences should start with empty token (as root of dependency tree)!

        tgt_heads, tgt_labels = tgt_seqs

        src_len = len(tgt_heads[0])
        batch_size = len(tgt_heads)
        np_tgt_heads = np.array(tgt_heads).flatten()  # (src_len * batch_size)
        np_tgt_labels = np.array(tgt_labels).flatten()
        s_arc, s_label = self.cal_scores(src_encodings)  # (src_len, src_len, bs), ([(src_len, src_len, bs)])

        s_arc_value = s_arc.npvalue()
        s_arc_choice = np.argmax(s_arc_value, axis=0).transpose().flatten()  # (src_len * batch_size)

        s_pick_labels = [dy.pick_batch(dy.reshape(score, (src_len,), batch_size=src_len * batch_size), s_arc_choice)
                         for score in s_label]
        s_argmax_labels = dy.concatenate(s_pick_labels, d=0)  # n_labels, src_len * batch_size

        reshape_s_arc = dy.reshape(s_arc, (src_len,), batch_size=src_len * batch_size)
        arc_loss = dy.pickneglogsoftmax_batch(reshape_s_arc, np_tgt_heads)
        label_loss = dy.pickneglogsoftmax_batch(s_argmax_labels, np_tgt_labels)

        loss = dy.sum_batches(arc_loss + label_loss) / batch_size
        return loss

    def decoding(self, src_encodings):
        src_len = len(src_encodings)

        # NOTE: should transpose before calling `mst` method!
        s_arc, s_label = self.cal_scores(src_encodings)
        s_arc_values = s_arc.npvalue().transpose()  # src_len, src_len
        s_label_values = np.asarray([x.npvalue() for x in s_label]) \
            .transpose((2, 1, 0))  # src_len, src_len, n_labels

        # weights = np.zeros((src_len + 1, src_len + 1))
        # weights[0, 1:(src_len + 1)] = np.inf
        # weights[1:(src_len + 1), 0] = np.inf
        # weights[1:(src_len + 1), 1:(src_len + 1)] = s_arc_values[batch]
        weights = s_arc_values
        pred_heads = mst(weights)
        pred_labels = [np.argmax(labels[head])
                       for head, labels in zip(pred_heads, s_label_values)]

        return pred_heads, pred_labels

    def cal_accuracy(self, pred_head, pred_labels, true_head, true_labels):
        head_acc = np.sum(np.equal(pred_head, true_head)) / len(pred_labels)
        label_acc = np.sum(np.equal(pred_labels, true_labels)) / len(pred_head)

        return head_acc, label_acc


class Max_Margin_Decoder(Decoder):
    def __init__(self, margin):
        Decoder.__init__(self)
        self.margin = margin

    def decode_loss(self, model, src_encoding, tgt_encoding):
        # The src encoding[i] and tgt encoding[i] are golden pairs
        src_encodings = dy.concatenate_cols(src_encoding)
        tgt_encodings = dy.concatenate_cols(tgt_encoding)

    def decode(self, src_encoding, tgt_encodings):
        # return a list of scores
        pass

