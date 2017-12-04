import codecs
from collections import defaultdict
# "GENERAL lookup table"
tags = set(['GPE', 'PER', 'ORG', 'LOC'])


def read_gold_file(gold_path):
    with codecs.open(gold_path, "r", "utf-8") as fin:
        doc_set = set()
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue

            line = line.decode('utf-8')
            tokens = line.split('\t')

            doc_id = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2])
            ner = tokens[5].split('/')[0]

            doc_set.add((doc_id, start, end))

        print 'num of annotated doc: %d' % len(doc_set)
    return doc_set


def make_darpa_format(span, curr_docum, curr_anot, start, end, tag):
    st = 'CMU_NER_LOREAL_CP1_TB_GS' + '\t' + curr_docum + '-ann-' + str(curr_anot) + '\t' + span\
    + '\t' + curr_docum + ':' + str(start) + '-' + str(end) + '\t' + 'NIL' + '\t' + \
    tag + '\t' + 'NAM' + '\t' + '1.0' + "\n"
    return st.split('\t')


def combine_lookup_table(lookup_files):
    lookup_table = defaultdict(lambda: set())

    for key, fname in lookup_files.iteritems():
        if key in tags:
            with codecs.open(fname, "r", "utf-8") as fin:
                for line in fin:
                    lookup_table[line.strip()].add(key)
        else:
            with codecs.open(fname, "r", "utf-8") as fin:
                for line in fin:
                    fs = line.strip().split('\t')
                    lookup_table[fs[0]].add(fs[1])
    new_lookup_table = dict()

    # remove spans that are annotated with multiple entities
    for key, value in lookup_table.iteritems():
        if len(value) == 1:
            new_lookup_table[key] = list(value)[0]
    return new_lookup_table


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


def post_processing(path_darpa_prediction,
                    path_to_full_setE,
                    path_to_author,
                    output_file,
                    lookup_files=None,
                    label_propagate=True,
                    conf_num=0,
                    gold_file_path=None):
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
    predicted_spans = defaultdict(lambda: list())

    if lookup_files is not None:
        lookup_table = combine_lookup_table(lookup_files)
    author_lookup = single_lookup_table(path_to_author, "PER")
    annot_id = defaultdict(lambda: 0) # doc_id:annotation num

    gold_spans = read_gold_file(gold_file_path)

    def _look_up(span, doc_attribute):
        if doc_attribute == "DF" and span in author_lookup:
            return 'PER'
        if lookup_files is not None and span in lookup_table:
            return lookup_table[span]
        return None

    def _is_overlap(s1, e1, s2, e2):
        # Condition: s1 < e1, s2 < e2
        return not(e1 < s2 or e2 < s1)

    def _check_cross_annotations(list_spans, target_start, target_end):
        flag = False
        for (s, e) in list_spans:
            if _is_overlap(s, e, target_start, target_end):
                flag = True
                break
        return flag

    add_labels = 0  # includes both fixed labels and added labels
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
            if lookup_tag is not None and lookup_tag != predict_tag and (doc_id, start_id, end_id) in gold_spans:
                add_labels += 1
            predict_tag = predict_tag if lookup_tag is None else lookup_tag

            predicted_doc[doc_id][(span, start_id, end_id)] = predict_tag
            prediction_list.append(make_darpa_format(span, doc_id, annot_id[doc_id], start_id, end_id, predict_tag))
            predicted_spans[doc_id].append((start_id, end_id))
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
            if len(tokens) == 0 or line == "" or line == "\n":
                ngrams, starts, ends = find_ngrams(one_sent, start_ids, end_ids, MAX_NGRAM)
                for ngram, s, e in zip(ngrams, starts, ends):
                    ngram = " ".join(ngram)
                    ngram_freq[ngram] += 1
                    predict_tag = _look_up(ngram, doc_attribute)
                    key = (ngram, s[0], e[-1])
                    if predict_tag is not None:
                        if key not in predicted_doc[doc_id] and not _check_cross_annotations(predicted_spans[doc_id], s[0], s[-1]):
                            predicted_doc[doc_id][key] = predict_tag
                            annot_id[doc_id] += 1
                            prediction_list.append(make_darpa_format(ngram, doc_id, annot_id[doc_id], s[0], e[-1], predict_tag))
                            predicted_spans[doc_id].append((s[0], e[-1]))
                            if (doc_id, s[0], e[-1]) in gold_spans:
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

    print("Total %d labels in the gold spans get fixed by the lookup tables!" % (add_labels,))

    tot_prop_label = 0
    if label_propagate:
        # Label propagation
        # (a) Within document propagation
        for doc_id, span_infos in predicted_doc.iteritems():
            vote_tag = defaultdict(lambda: defaultdict(list))  # span: tag:[(start, end)]
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
                new_vote_tag[span] = (max_tag, vote_tag[span][max_tag], max_vote)

            add_label = 0
            for unpredict_span in unpredicted_spans[doc_id]:
                s2, e2 = unpredict_span[1], unpredict_span[2]
                uspan = unpredict_span[0]
                if uspan in new_vote_tag:
                    # conservative propagation
                    if new_vote_tag[uspan][2] <= conf_num:
                        continue
                    pred_tag = new_vote_tag[uspan][0]
                    # check if there is an overlap between spans
                    flag = True
                    for s1, e1 in new_vote_tag[uspan][1]:
                        if _is_overlap(s1, e1, s2, e2):
                            print "There is overlap: ", (s1, e1), (s2, e2)
                            flag = False
                            break
                    if flag and not _check_cross_annotations(predicted_spans[doc_id], s2, e2):
                        # propagate the label
                        if (doc_id, s2, e2) in gold_spans:
                            add_label += 1
                        annot_id[doc_id] += 1
                        prediction_list.append(make_darpa_format(uspan, doc_id, annot_id[doc_id], s2, e2, pred_tag))
                        predicted_spans[doc_id].append((s2, e2))
                        unpredicted_spans[doc_id].remove(unpredict_span)
            if add_label > 0:
                tot_prop_label += add_label
                print("Within Document Label Propagation: Add %d labels for Doc %s. " % (add_label, doc_id))

        print("Total %d labels get propagated within document for gold setE!" % (tot_prop_label, ))

        # (b) Cross document propagation
        most_freq_num = 20
        freq_ngram_list = sorted(ngram_freq, key=ngram_freq.get)[-most_freq_num:]
        for w in freq_ngram_list:
            print w
        vote_tag = defaultdict(lambda: defaultdict(lambda :0))
        for doc_id, span_infos in predicted_doc.iteritems():
            for span_info, tag in span_infos.iteritems():
                span = span_info[0]
                if span in freq_ngram_list:
                    vote_tag[span][tag] += 1
                    print span
        vote_out_ents = dict()
        for span, other in vote_tag.iteritems():
            max_tag = ""
            max_vote = 0
            for tag, vote in other.iteritems():
                if vote > max_vote:
                    max_tag = tag
                    max_vote = vote
            vote_out_ents[span] = max_tag
        print "voted entities:", vote_out_ents
        add_label = 0
        for doc_id, unpredict_span_list in unpredicted_spans.iteritems():
            for unpredict_span in unpredict_span_list:
                start, end = unpredict_span[1], unpredict_span[2]
                uspan = unpredict_span[0]
                if uspan in vote_out_ents and not _check_cross_annotations(predicted_spans[doc_id], start, end):
                    if (doc_id, s2, e2) in gold_spans:
                        add_label += 1
                    annot_id[doc_id] += 1
                    prediction_list.append(
                        make_darpa_format(uspan, doc_id, annot_id[doc_id], start, end, vote_out_ents[uspan]))
                    predicted_spans[doc_id].append((start, end))
                    unpredicted_spans[doc_id].remove(unpredict_span)
        print("Total %d labels get propagated across document for gold setE!" % (add_label, ))

    with codecs.open(output_file, "w", encoding='utf-8') as fout:
        for item in prediction_list:
            one_sent = "\t".join(item)
            fout.write(one_sent)


if __name__ == "__main__":
    author_list = "../eval/oromo/set0E_author.txt"
    author_list = "./debug/set012E_author.txt"
    setE_conll = "../new_datasets/setE/oromo/setE.conll"
    pred = "./debug/pred.conll"
    # pred = "./post_test.txt"
    # lookup_file = {"Gen": "../eval/oromo/Oromo_Annotated.txt"}
    output_file = "post_test.txt"
    gold_file_path = "../ner_score/orm_setE_edl.tac"
    post_processing(pred, setE_conll, author_list, output_file, lookup_files=None, label_propagate=True, gold_file_path=gold_file_path)
    # post_process_lookup(pred, setE_conll, author_list, output_file, lookup_file)

