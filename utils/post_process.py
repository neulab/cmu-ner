
import codecs
from collections import defaultdict
# "GENERAL lookup table"
tags = set(['GPE', 'PER', 'ORG', 'LOC'])


def make_darpa_format(span, curr_docum, curr_anot, start, end, tag):
    st = 'CMU_NER_LOREAL_CP1_TB_GS' + '\t' + curr_docum + '-ann-' + str(curr_anot) + '\t' + span\
    + '\t' + curr_docum + ':' + str(start) + '-' + str(end) + '\t' + 'NIL' + '\t' + \
    tag + '\t' + 'NAM' + '\t' + '1.0' + "\n"
    return st.split('\t')


def combine_lookup_table(lookup_files):
    lookup_table = dict()

    for key, fname in lookup_files.iteritems():
        if key in tags:
            with codecs.open(fname, "r", "utf-8") as fin:
                for line in fin:
                    lookup_table[line.strip()] = key
        else:
            with codecs.open(fname, "r", "utf-8") as fin:
                for line in fin:
                    fs = line.strip().split('\t')
                    lookup_table[fs[0]] = fs[1]
    return lookup_table


def single_lookup_table(lookup_file, tag):
    lookup_table = dict()
    if tag in tags:
        with codecs.open(lookup_file, "r", "utf-8") as fin:
            for line in fin:
                lookup_table[line.strip()] = tag
    else:
        with codecs.open(lookup_file, "r", "utf-8") as fin:
            for line in fin:
                fs = line.strip().split('\t')
                lookup_table[fs[0]] = fs[1]
    return lookup_table


def find_ngrams(sent, starts, ends, n):
    all_ngrams = []
    all_starts = []
    all_ends = []
    for i in range(1, n+1):
        all_ngrams += zip(*[sent[j:] for j in range(i)])
        all_starts += zip(*[starts[j:] for j in range(i)])
        all_ends += zip(*[ends[j:] for j in range(i)])
    return all_ngrams, all_starts, all_ends


def post_process(path_darpa_prediction, path_to_full_setE, path_to_author, output_file, lookup_files=None, label_propagate=True):
    '''

    :param path_darpa_prediction: Final output
    :param path_to_full_setE: setE.conll
    :param path_to_author: "path_to_author_list"
    :param output_file:
    :param lookup_files: {"General": "path_to_lexicon_1", "General": path2"}
    :param label_propagate: BOOLEAN
    :return:
    '''
    predicted_doc = defaultdict(lambda: dict()) # (doc_id: (span_token, start, end):NER)
    unpredicted_spans = defaultdict(lambda: list()) # (doc_id: [(ngram_token, start, end)])
    MAX_NGRAM = 5
    prediction_list = []
    if lookup_files is not None:
        lookup_table = combine_lookup_table(lookup_files)
    author_lookup = single_lookup_table(path_to_author, "PER")
    annot_id = defaultdict(lambda: 0) # doc_id:annotation num

    def _look_up(span, doc_attribute):
        if doc_attribute == "DF" and span in author_lookup:
            return 'PER'
        if lookup_files is not None and span in lookup_table:
            return lookup_table[span]
        return None

    # First using the lookup table to fix up the current predictions
    with codecs.open(path_darpa_prediction, "r", "utf-8") as fin:
        for line in fin:
            fields = line.strip().split('\t')
            span = fields[2]
            predict_tag = fields[5]
            doc_id_span = fields[3].split(":")
            doc_id = doc_id_span[0]
            doc_attribute = doc_id.split('_')[1]
            annot_id[doc_id] += 1
            span_id = [int(i.strip()) for i in doc_id_span[1].split('-')]
            start_id, end_id = span_id[0], span_id[1]

            lookup_tag = _look_up(span, doc_attribute)
            predict_tag = predict_tag if lookup_tag is None else lookup_tag

            predicted_doc[doc_id][(span, start_id, end_id)] = predict_tag
            prediction_list.append(make_darpa_format(span, doc_id, annot_id[doc_id], start_id, end_id, predict_tag))

    # Second, iterate over the full setE using the lookup tables to completed the predicted dict
    # In the mean time, give statistics of ngrams for label propagation.
    ngram_freq = defaultdict(lambda: 0)
    with codecs.open(path_to_full_setE, "r", "utf-8") as fin:
        one_sent = []
        start_ids = []
        end_ids = []
        doc_attribute = ""
        for line in fin:
            tokens = line.split('\t')
            if line == "" or line == "\n":
                ngrams, starts, ends = find_ngrams(one_sent, start_ids, end_ids, MAX_NGRAM)
                for ngram, s, e in zip(ngrams, starts, ends):
                    ngram = " ".join(ngram)
                    ngram_freq[ngram] += 1
                    predict_tag = _look_up(ngram, doc_attribute)
                    key = (ngram, s[0], e[-1])
                    if predict_tag is not None:
                        if key not in predicted_doc[doc_id]:
                            predicted_doc[doc_id][key] = predict_tag
                            annot_id[doc_id] += 1
                            prediction_list.append(make_darpa_format(ngram, doc_id, annot_id[doc_id], s[0], e[-1], predict_tag))
                    else:
                        if key not in predicted_doc[doc_id]:
                            unpredicted_spans[doc_id].append(key)
                one_sent = []
                start_ids = []
                end_ids = []
            else:
                word = tokens[0]
                doc_id = tokens[3]
                doc_attribute = doc_id.split('_')[1]
                start = int(tokens[6])
                end = int(tokens[7])

                one_sent.append(word)
                start_ids.append(start)
                end_ids.append(end)

    def _is_overlap(s1, e1, s2, e2):
        # Condition: s1 < e1, s2 < e2
        return not(e1 < s2 or e2 < s1)

    # Label propagation
    # (a) Within document propagation
    for doc_id, span_infos in predicted_doc.iteritems():
        vote_tag = defaultdict(lambda: defaultdict(list())) # span: tag:[(start, end)]
        for span_info, tag in span_infos.iteritems():
            span = span_info[0]
            start = span_info[1]
            end = span_info[2]
            vote_tag[span][tag].append((start, end))
        new_vote_tag = dict()
        for span, other in vote_tag.iteritems():
            max_tag = ""
            max_vote = 0
            for tag in other.keys():
                vote = len(other[tag])
                if vote > max_vote:
                    max_vote = vote
                    max_tag = tag
            new_vote_tag[span] = (max_tag, vote_tag[span][max_tag])

        add_label = 0
        for unpredict_span in unpredicted_spans[doc_id]:
            s2, e2 = unpredict_span[1], unpredict_span[2]
            uspan = unpredict_span[0]
            if uspan in new_vote_tag:
                pred_tag = new_vote_tag[uspan][0]
                # check if there is an overlap between spans
                flag = True
                for s1, e1 in new_vote_tag[uspan][1]:
                    if _is_overlap(s1, e1, s2, e2):
                        print "There is overlap: ", (s1, e1), (s2, e2)
                        flag = False
                        break
                if flag:
                    # propagate the label
                    add_label += 1
                    annot_id[doc_id] += 1
                    prediction_list.append(make_darpa_format(uspan, doc_id, annot_id[doc_id], s2, e2, pred_tag))
                    unpredicted_spans[doc_id].remove(unpredict_span)
        print("Within Document Label Propagation: Add %d labels for Doc %s. " % (add_label, doc_id))

    with codecs.open(output_file, "w", encoding='utf-8') as fout:
        for item in prediction_list:
            one_sent = "\t".join(item)
            fout.write(one_sent + "\n")


def post_process_lookup(path_darpa_prediction, path_to_full_setE, path_to_author, output_file, lookup_files=None):
    predicted_doc = defaultdict(lambda: dict()) # (doc_id: (span_token, start, end):NER)
    unpredicted_spans = defaultdict(lambda: list()) # (doc_id: [(ngram_token, start, end)])
    MAX_NGRAM = 5
    prediction_list = []
    if lookup_files is not None:
        lookup_table = combine_lookup_table(lookup_files)
    author_lookup = single_lookup_table(path_to_author, "PER")
    annot_id = defaultdict(lambda: 0) # doc_id:annotation num

    def _look_up(span, doc_attribute):
        if doc_attribute == "DF" and span in author_lookup:
            return 'PER'
        if lookup_files is not None and span in lookup_table:
            return lookup_table[span]
        return None

    add_labels = 0 # includes both fixed labels and added labels
    # First using the lookup table to fix up the current predictions
    with codecs.open(path_darpa_prediction, "r", "utf-8") as fin:
        for line in fin:
            fields = line.strip().split('\t')
            span = fields[2]
            predict_tag = fields[5]
            doc_id_span = fields[3].split(":")
            doc_id = doc_id_span[0]
            doc_attribute = doc_id.split('_')[1]
            annot_id[doc_id] += 1
            span_id = [int(i.strip()) for i in doc_id_span[1].split('-')]
            start_id, end_id = span_id[0], span_id[1]

            lookup_tag = _look_up(span, doc_attribute)
            predict_tag = predict_tag if lookup_tag is None else lookup_tag

            predicted_doc[doc_id][(span, start_id, end_id)] = predict_tag
            prediction_list.append(make_darpa_format(span, doc_id, annot_id[doc_id], start_id, end_id, predict_tag))

    # Second, iterate over the full setE using the lookup tables to completed the predicted dict
    # In the mean time, give statistics of ngrams for label propagation.
    ngram_freq = defaultdict(lambda: 0)

    with codecs.open(path_to_full_setE, "r", "utf-8") as fin:
        one_sent = []
        start_ids = []
        end_ids = []
        doc_attribute = ""

        for line in fin:
            tokens = line.strip().split('\t')
            if line == "" or line == "\n":
                ngrams, starts, ends = find_ngrams(one_sent, start_ids, end_ids, MAX_NGRAM)
                for ngram, s, e in zip(ngrams, starts, ends):
                    ngram = " ".join(ngram)
                    ngram_freq[ngram] += 1
                    predict_tag = _look_up(ngram, doc_attribute)
                    key = (ngram, s[0], e[-1])
                    if predict_tag is not None:
                        if key not in predicted_doc[doc_id]:
                            predicted_doc[doc_id][key] = predict_tag
                            annot_id[doc_id] += 1
                            prediction_list.append(make_darpa_format(ngram, doc_id, annot_id[doc_id], s[0], e[-1], predict_tag))
                            add_labels += 1
                    else:
                        if key not in predicted_doc[doc_id]:
                            unpredicted_spans[doc_id].append(key)
                one_sent = []
                start_ids = []
                end_ids = []
            else:
                word = tokens[0]
                doc_id = tokens[3]
                doc_attribute = doc_id.split('_')[1]
                start = int(tokens[6])
                end = int(tokens[7])

                one_sent.append(word)
                start_ids.append(start)
                end_ids.append(end)

    print("Total %d labels get fixed!" % (add_labels, ))
    with codecs.open(output_file, "w", encoding='utf-8') as fout:
        for item in prediction_list:
            one_sent = "\t".join(item)
            fout.write(one_sent)

if __name__ == "__main__":
    author_list = "../eval/oromo/set0E_author.txt"
    setE_conll = "../eval/oromo/setE.conll"
    pred = "../eval/oromo/cp1_orm_som_trans_0.015_500_somTEmb_8bc874_darpa_output.conll"
    lookup_file = {"Gen": "../eval/oromo/lexicon_annoatated.txt"}
    output_file = "post_test.txt"
    # post_process(pred, setE_conll, author_list, output_file, lookup_file)
    post_process_lookup(pred, setE_conll, author_list, output_file, lookup_file)

