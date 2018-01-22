__author__ = 'chuntingzhou'
from utils.util import *


class Decoder():
    def __init__(self, tag_size):
        # type: () -> object
        pass

    def decode_loss(self):
        raise NotImplementedError

    def decoding(self):
        raise NotImplementedError


def constrained_transition_init(transition_matrix, contraints):
    '''
    :param transition_matrix: numpy array, (from, to)
    :param contraints: [[from_indexes], [to_indexes]]
    :return: newly initialized transition matrix
    '''
    for cons in contraints:
        transition_matrix[cons[0], cons[1]] = -1000.0
    return transition_matrix


class chain_CRF_decoder(Decoder):
    ''' For NER and POS Tagging. '''

    def __init__(self, model, src_output_dim, tag_emb_dim, tag_size, constraints=None):
        Decoder.__init__(self, tag_size)
        self.model = model
        self.start_id = tag_size
        self.end_id = tag_size + 1
        self.tag_size = tag_size + 2
        tag_size = tag_size + 2

        # optional: transform the hidden space of src encodings into the tag embedding space
        self.W_src2tag_readout = model.add_parameters((tag_emb_dim, src_output_dim))
        self.b_src2tag_readout = model.add_parameters((tag_emb_dim))
        self.b_src2tag_readout.zero()

        self.W_scores_readout2tag = model.add_parameters((tag_size, tag_emb_dim))
        self.b_scores_readout2tag = model.add_parameters((tag_size))
        self.b_scores_readout2tag.zero()

        # (to, from), trans[i] is the transition score to i
        init_transition_matrix = np.random.randn(tag_size, tag_size)
        init_transition_matrix[self.end_id, :] = -1000.0
        init_transition_matrix[:, self.start_id] = -1000.0
        if constraints is not None:
            init_transition_matrix = constrained_transition_init(init_transition_matrix, constraints)
        # print init_transition_matrix
        self.transition_matrix = model.add_lookup_parameters((tag_size, tag_size),
                                                             init=dy.NumpyInitializer(init_transition_matrix))


    def forward_alg(self, tag_scores, batch_size):
        ''' Forward DP for CRF..
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        # Be aware: if a is lookup_parameter with 2 dimension, then a[i] returns one row;
        # if b = dy.parameter(a), then b[i] returns one column; which means dy.parameter(a) already transpose a
        transpose_transition_score = dy.parameter(self.transition_matrix)
        # transpose_transition_score = dy.transpose(transition_score)  # (from, to)
        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        # np_init_alphas = -100.0 * np.ones((self.tag_size, batch_size))
        # np_init_alphas[self.start_id, :] = 0.0
        # alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)
        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]

        for tag_score in tag_scores[1:]:
            # extend for each transit <to>
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)

        terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
        return terminal_alpha

    def score_one_sequence(self, tag_scores, tags, batch_size):
        ''' tags: list of tag ids at each time step '''
        # print tags, batch_size
        # print batch_size
        # print "scoring one sentence"
        tags = [[self.start_id] * batch_size] + tags  # len(tag_scores) = len(tags) - 1
        score = dy.inputTensor(np.zeros(batch_size), batched=True)
        # tag_scores = dy.concatenate_cols(tag_scores) # tot_tags, sent_len, batch_size
        # print "tag dim: ", tag_scores.dim()
        for i in range(len(tags) - 1):
            score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, tags[i + 1]), tags[i]) \
                    + dy.pick_batch(tag_scores[i], tags[i + 1])
        score += dy.pick_batch(dy.lookup_batch(self.transition_matrix, [self.end_id]*batch_size), tags[-1])
        return score

    def score_one_sequence_partial(self, tag_scores, tags, batch_size, known_tags, tag_to_id, B_UNK_tag, I_UNK_tag):

        transpose_transition_score = dy.parameter(self.transition_matrix)
        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]

        # Make mask for first tag
        mask_w_0_all_s = self.makeMask(batch_size, known_tags, tag_to_id, B_UNK_tag, I_UNK_tag, tags, 0)
        i = 1
        #Adding tags to the <from start_tag> to the <to> tags
        #alpha_tm1 = dy.cmult(alpha_tm1  , dy.inputTensor(mask_w_0_all_s, batched=True))
        alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_0_all_s, batched=True)


        for tag_score in tag_scores[1:]:

            # extend for each transit <to>
            alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)

            # each column i of tag_score will be the repeated emission score to tag i
            tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))

            alpha_t = alpha_tm1 + transpose_transition_score + tag_score
            alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)

            #extracting masks for <from> tag
            mask_w_i_all_s = self.makeMask(batch_size, known_tags, tag_to_id, B_UNK_tag, I_UNK_tag, tags, i)
            #alpha_tm1 = dy.cmult(alpha_tm1 ,  dy.inputTensor(mask_w_i_all_s, batched=True))
            alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_i_all_s, batched=True)
            i = i + 1


        terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
        return terminal_alpha

    def makeMask(self, batch_size, known_tags, tag_to_id, B_UNK_tag, I_UNK_tag, tags, index):
        # mask_w_0 = dy.transpose(dy.concatenate_cols([dy.scalarInput(-2000)] * self.tag_size))
        # mask_w_0_all_s = dy.concatenate_cols([mask_w_0] * batch_size)
        mask_w_0 = np.array([[-1000] * self.tag_size])
        mask_w_0 = np.transpose(mask_w_0)
        mask_w_0_all_s  = np.reshape(np.array([mask_w_0] * batch_size), (self.tag_size,batch_size))

        mask_idx = []
        tag_vals = []
        for idx, w0_si in enumerate(known_tags[index]):
            if w0_si[0] == 1:
                mask_idx.append(idx)
                tag_vals.append(tags[index][idx])
            else:
                if tags[index][idx] == B_UNK_tag:  # Possible labels it can take are B_LOC, B_PER, B_ORG, B_GPE, O
                    possible_labels = ["B-LOC", "B-PER", "B-ORG", "B-GPE", "O"]
                elif tags[index][idx] == I_UNK_tag:
                    possible_labels = ["I-LOC", "I-PER", "I-ORG", "I-GPE", "O"]

                for pl in possible_labels:
                    mask_idx.append(idx)
                    tag_vals.append(tag_to_id[pl])
        mask_w_0_all_s[tag_vals,  mask_idx] = 0
        return mask_w_0_all_s


    def generateMaskForToTags(self, batch_size, known_tags, tag_to_id, B_UNK_tag, I_UNK_tag, tags, index, to_tag):
        mask_idx = []
        tag_values = []
        #Creating a mask for each <to> tag
        mask_w_i = np.array([[0] * self.tag_size])
        mask_w_i = np.transpose(mask_w_i)
        mask_w_i_all_s = np.reshape(np.array([mask_w_i] * batch_size), (self.tag_size, batch_size))

        for idx, wi_si in enumerate(known_tags[index]):
            if wi_si[0] == 1 and tags[index][idx] == to_tag: #The tag is known for a sentence i, then add the index of the sentence to maskId
                mask_idx.append(idx)
                tag_values.append(to_tag)
            elif wi_si[0] == 0:#If the tag is UNK, then we need to try for all the tags
                if tags[index][idx] == B_UNK_tag:
                    possible_labels = [tag_to_id["B-LOC"], tag_to_id["B-PER"], tag_to_id["B-ORG"], tag_to_id["B-GPE"], tag_to_id["O"]]
                elif tags[index][idx] == I_UNK_tag:
                        possible_labels = [tag_to_id["I-LOC"], tag_to_id["I-PER"], tag_to_id["I-ORG"],tag_to_id["I-GPE"], tag_to_id["O"]]
                if to_tag in possible_labels:
                    mask_idx.append(idx)
                    tag_values.append(to_tag)


        mask_w_i_all_s[tag_values, mask_idx] = 1

        return mask_w_i_all_s


    def decode_loss(self, src_encodings, tgt_tags, known, use_partial, tag_to_id, B_UNK_tag, I_UNK_tag):
        # This is the batched version which requires bucketed batch input with the same length.
        '''
        The length of src_encodings and tgt_tags are time_steps.
        src_encodings: list of dynet.Tensor (src_output_dim, batch_size)
        tgt_tags: list of tag ids [(1, batch_size)]
        return: average of negative log likelihood
        '''
        # TODO: transpose tgt tags first
        batch_size = len(tgt_tags)
        tgt_tags, tgt_mask = transpose_input(tgt_tags, 0)
        known_tags, _ = transpose_input(known, 0)
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding])) for src_encoding
                    in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        # scores over all paths, all scores are in log-space
        forward_scores = self.forward_alg(tag_scores, batch_size)
        if use_partial:
            gold_score = self.score_one_sequence_partial(tag_scores, tgt_tags, batch_size, known_tags,tag_to_id, B_UNK_tag, I_UNK_tag)
        else:
            gold_score = self.score_one_sequence(tag_scores, tgt_tags, batch_size)
        # negative log likelihood
        loss = dy.sum_batches(forward_scores - gold_score) / batch_size
        return loss #, dy.sum_batches(forward_scores)/batch_size, dy.sum_batches(gold_score) / batch_size

    def get_crf_scores(self, src_encodings):
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding]))
                    for src_encoding in src_encodings]
        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        transpose_transition_score = dy.parameter(self.transition_matrix)  # (to, from)

        return transpose_transition_score.npvalue(), [ts.npvalue() for ts in tag_scores]

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
        np_init_alpha = np.ones(self.tag_size) * -2000.0
        np_init_alpha[self.start_id] = 0.0
        max_tm1 = dy.inputTensor(np_init_alpha)
        transpose_transition_score = dy.parameter(self.transition_matrix)  # (to, from)

        for i, tag_score in enumerate(tag_scores):
            max_tm1 = dy.concatenate_cols([max_tm1] * self.tag_size)
            max_t = max_tm1 + transpose_transition_score
            if i != 0:
                eval_score = max_t.npvalue()[:-2, :]
            else:
                eval_score = max_t.npvalue()
            best_tag = np.argmax(eval_score, axis=0)
            back_trace_tags.append(best_tag)
            max_tm1 = dy.inputTensor(eval_score[best_tag, range(self.tag_size)]) + tag_score

        terminal_max_T = max_tm1 + self.transition_matrix[self.end_id]
        eval_terminal = terminal_max_T.npvalue()[:-2]
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
        return np.sum(np.equal(pred_path, true_path).astype(np.float32)) / len(pred_path)


def ensemble_viterbi_decoding(l_tag_scores, l_transit_score, tag_size):
    back_trace_tags = []
    tag_size = tag_size + 2
    start_id = tag_size - 2
    end_id = tag_size - 1
    max_tm1 = np.ones(tag_size) * -2000.0
    max_tm1[start_id] = 0.0

    tag_scores = []
    for i in range(len(l_tag_scores[0])):
        tag_scores.append(sum([ts[i] for ts in l_tag_scores]) / len(l_tag_scores))
    transpose_transition_score = sum(l_transit_score) / len(l_transit_score)  # (from, to)

    for i, tag_score in enumerate(tag_scores):
        max_tm1 = np.tile(np.expand_dims(max_tm1, axis=1), (1, tag_size))
        max_t = max_tm1 + transpose_transition_score
        if i != 0:
            eval_score = max_t[:-2, :]
        else:
            eval_score = max_t
        best_tag = np.argmax(eval_score, axis=0)
        back_trace_tags.append(best_tag)
        max_tm1 = eval_score[best_tag, range(tag_size)] + tag_score

    terminal_max_T = max_tm1 + transpose_transition_score[:, end_id]
    eval_terminal = terminal_max_T[:-2]
    best_tag = np.argmax(eval_terminal, axis=0)
    best_path_score = eval_terminal[best_tag]

    best_path = [best_tag]
    for btpoint in reversed(back_trace_tags):
        best_tag = btpoint[best_tag]
        best_path.append(best_tag)
    start = best_path.pop()
    assert start == start_id
    best_path.reverse()
    return best_path_score, best_path


class classifier(Decoder):
    def __init__(self, model, input_dim, tag_size):
        self.W_softmax = model.add_parameters((tag_size, input_dim))
        self.b_softmax = model.add_parameters((tag_size))

    def decode_loss(self, src_encoding, tgt_tags):
        batch_size = len(tgt_tags)
        tgt_tags, tgt_mask = transpose_input(tgt_tags, 0)

        assert len(src_encoding) == len(tgt_tags)

        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)

        predictions = [dy.affine_transform([b_softmax, W_softmax, src_emb]) for src_emb in src_encoding]

        losses = [dy.pickneglogsoftmax_batch(pred, tgt) for pred, tgt in zip(predictions, tgt_tags)]

        loss = dy.sum_batches(dy.esum(losses)) / (batch_size * len(src_encoding))

        return loss

    def decoding(self, src_encoding):
        W_softmax = dy.parameter(self.W_softmax)
        b_softmax = dy.parameter(self.b_softmax)
        predictions = [dy.affine_transform([b_softmax, W_softmax, src_emb]) for src_emb in src_encoding]

        predictions = [np.argmax(pred.npvalue()) for pred in predictions]

        return None, predictions
