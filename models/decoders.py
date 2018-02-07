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

    def __init__(self, args, model, src_output_dim, tag_emb_dim, tag_size, constraints=None):
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
        init_transition_matrix = np.random.randn(tag_size, tag_size) # from, to
        # init_transition_matrix[self.start_id, :] = -1000.0
        # init_transition_matrix[:, self.end_id] = -1000.0
        init_transition_matrix[self.end_id, :] = -1000.0
        init_transition_matrix[:, self.start_id] = -1000.0
        if constraints is not None:
            init_transition_matrix = constrained_transition_init(init_transition_matrix, constraints)
        # print init_transition_matrix
        self.transition_matrix = model.add_lookup_parameters((tag_size, tag_size),
                                                             init=dy.NumpyInitializer(init_transition_matrix))

        self.interpolation = args.interp_crf_score
        if self.interpolation:
            self.W_weight_transition = model.add_parameters((1, tag_emb_dim))
            self.b_weight_transition = model.add_parameters((1))
            self.b_weight_transition.zero()

    def forward_alg(self, tag_scores):
        ''' Forward DP for CRF.
        tag_scores (list of batched dy.Tensor): (tag_size, batchsize)
        '''
        # Be aware: if a is lookup_parameter with 2 dimension, then a[i] returns one row;
        # if b = dy.parameter(a), then b[i] returns one column; which means dy.parameter(a) already transpose a
        transpose_transition_score = dy.parameter(self.transition_matrix)
        # transpose_transition_score = dy.transpose(transition_score)
        # alpha(t', s) = the score of sequence from t=0 to t=t' in log space
        # np_init_alphas = -100.0 * np.ones((self.tag_size, batch_size))
        # np_init_alphas[self.start_id, :] = 0.0
        # alpha_tm1 = dy.inputTensor(np_init_alphas, batched=True)

        alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]
        # self.transition_matrix[i]: from i, column
        # transpose_score[i]: to i, row
        # transpose_score: to, from

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

    def decode_loss(self, src_encodings, tgt_tags,use_partial, known_tags, tag_to_id, B_UNK, I_UNK):
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
	known_tags, _ = transpose_input(known_tags, 0)
        W_src2tag_readout = dy.parameter(self.W_src2tag_readout)
        b_src2tag_readout = dy.parameter(self.b_src2tag_readout)
        W_score_tag = dy.parameter(self.W_scores_readout2tag)
        b_score_tag = dy.parameter(self.b_scores_readout2tag)

        tag_embs = [dy.tanh(dy.affine_transform([b_src2tag_readout, W_src2tag_readout, src_encoding])) for src_encoding
                    in src_encodings]
        if self.interpolation:
            W_transit = dy.parameter(self.W_weight_transition)
            b_transit = dy.parameter(self.b_weight_transition)
            step_weight_on_transit = [dy.logistic(dy.affine_transform([b_transit, W_transit, tag_emb])) for tag_emb in tag_embs]

        tag_scores = [dy.affine_transform([b_score_tag, W_score_tag, tag_emb]) for tag_emb in tag_embs]

        # scores over all paths, all scores are in log-space
        forward_scores = self.forward_alg(tag_scores)
	if use_partial:
	    gold_score = self.score_one_sequence_partial(tag_scores, tgt_tags, batch_size,known_tags, tag_to_id, B_UNK, I_UNK)
	else:
	    gold_score = self.score_one_sequence(tag_scores, tgt_tags, batch_size)
        # negative log likelihood
        loss = dy.sum_batches(forward_scores - gold_score) / batch_size
        return loss #, dy.sum_batches(forward_scores)/batch_size, dy.sum_batches(gold_score) / batch_size
    
    def makeMask(self, batch_size, known_tags, tag_to_id, tags, index, B_UNK, I_UNK):
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
		if tags[index][idx] == B_UNK:
		    possible_labels = ["B-LOC", "B-PER", "B-ORG", "B-MISC", "O"]
		elif tags[index][idx] == I_UNK:
		    possible_labels = ["I-LOC", "I-PER", "I-ORG", "I-MISC", "O"]

		for pl in possible_labels:
		    mask_idx.append(idx)
		    tag_vals.append(tag_to_id[pl])
	mask_w_0_all_s[tag_vals,  mask_idx] = 0
	return mask_w_0_all_s

    def score_one_sequence_partial(self, tag_scores, tags, batch_size, known_tags, tag_to_id, B_UNK, I_UNK):
	transpose_transition_score = dy.parameter(self.transition_matrix)
	alpha_tm1 = transpose_transition_score[self.start_id] + tag_scores[0]

	mask_w_0_all_s = self.makeMask(batch_size, known_tags, tag_to_id, tags, 0, B_UNK, I_UNK)
	i = 1
	alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_0_all_s, batched=True)
	for tag_score in tag_scores[1:]:
	    alpha_tm1 = dy.concatenate_cols([alpha_tm1] * self.tag_size)  # (from, to, batch_size)
	    tag_score = dy.transpose(dy.concatenate_cols([tag_score] * self.tag_size))
	    alpha_t = alpha_tm1 + transpose_transition_score + tag_score
	    alpha_tm1 = log_sum_exp_dim_0(alpha_t)  # (tag_size, batch_size)
	    mask_w_i_all_s = self.makeMask(batch_size, known_tags, tag_to_id,tags, i,B_UNK, I_UNK)
	    alpha_tm1 = alpha_tm1 + dy.inputTensor(mask_w_i_all_s, batched=True)
	    i = i + 1
	
	terminal_alpha = log_sum_exp_dim_0(alpha_tm1 + self.transition_matrix[self.end_id])  # (1, batch_size)
	return terminal_alpha


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
