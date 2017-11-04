__author__ = 'chuntingzhou'
import argparse
from dataloaders.data_loader import *
from models.model_builder import *
import os
import uuid
from models.Convert_Output_Darpa import *
from models.Convert_to_darpa_xml import *

uid = uuid.uuid4().get_hex()[:6]


def evaluate(data_loader, path, model):
    sents, char_sents, tgt_tags, discrete_features = data_loader.get_data_set(path, args.lang, training=False)

    # tot_acc = 0.0
    predictions = []
    gold_standards = []
    i = 0
    for sent, char_sent, tgt_tag, discrete_feature in zip(sents, char_sents, tgt_tags, discrete_features):
        sent, char_sent, discrete_feature = [sent], [char_sent], [discrete_feature]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature, training=False)

        assert len(best_path) == len(tgt_tag)
        # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
        # tot_acc += acc
        predictions.append(best_path)
        gold_standards.append(tgt_tag)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.txt" % (str(uid))
    eval_output_fname = "%s_eval_score.txt" % (str(uid))
    with open(pred_output_fname, "w") as fout:
        for pred, gold in zip(predictions, gold_standards):
            for p, g in zip(pred, gold):
                fout.write("XXX " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    os.system("../eval/conlleval.v2 < %s > %s" % (pred_output_fname, eval_output_fname))

    with open(eval_output_fname, "r") as fin:
        lid = 0
        for line in fin:
            if lid == 1:
                fields = line.split(";")
                acc = float(fields[0].split(":")[1].strip()[:-1])
                precision = float(fields[1].split(":")[1].strip()[:-1])
                recall = float(fields[2].split(":")[1].strip()[:-1])
                f1 = float(fields[3].split(":")[1].strip())
            lid += 1

    output = open(eval_output_fname, "r").read().strip()
    print output
    os.system("rm %s" % (eval_output_fname))
    os.system("rm %s" % (pred_output_fname))

    return acc, precision, recall, f1


def evaluate_lr(data_loader, path, model):
    sents, char_sents, discrete_features, origin_sents = data_loader.get_lr_test(path, args.lang)

    predictions = []
    i = 0
    for sent, char_sent, discrete_feature in zip(sents, char_sents, discrete_features):
        sent, char_sent, discrete_feature = [sent], [char_sent], [discrete_feature]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature,training=False)

        predictions.append(best_path)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    pred_output_fname = "../eval/%s_pred_output.conll" % (str(uid))
    with codecs.open(pred_output_fname, "w", encoding='utf-8') as fout:
        for pred, sent in zip(predictions, origin_sents):
            for p, word in zip(pred, sent):
                fout.write(word + "\tNNP\tNP\t" + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    pred_darpa_output_fname = "../eval/%s_darpa_pred_output.conll" % (str(uid))
    final_darpa_output_fname = "../eval/%s_darpa_output.conll" % (str(uid))
    run_program(pred_output_fname, pred_darpa_output_fname, args.setEconll)

    run_program_darpa(pred_darpa_output_fname, final_darpa_output_fname)

    os.system("bash ../../ner_score/score_tig.sh ../eval/%s" % (final_darpa_output_fname))

    with codecs.open(final_darpa_output_fname,'r') as fileout:
        for line in fileout:
            columns = line.strip().split('\t')
            if len(columns) == 8 and columns[-1] == "strong_typed_mention_match":
                prec= columns[-4]
                recall = columns[-3]
                f1 = columns[-2]
                break

    return 0, prec, recall, f1


def main(args):
    ner_data_loader = NER_DataLoader(args)

    print ner_data_loader.id_to_tag

    if not args.data_aug:
        sents, char_sents, tgt_tags, discrete_features = ner_data_loader.get_data_set(args.train_path, args.lang)
    else:
        sents_tgt, char_sents_tgt, tags_tgt, dfs_tgt = ner_data_loader.get_data_set(args.tgt_lang_train_path, args.lang)
        sents_aug, char_sents_aug, tags_aug, dfs_aug = ner_data_loader.get_data_set(args.aug_lang_train_path, args.aug_lang)
        sents, char_sents, tgt_tags, discrete_features = sents_tgt+sents_aug, char_sents_tgt+char_sents_aug, tags_tgt+tags_aug, dfs_tgt+dfs_aug

    print ner_data_loader.char_to_id

    epoch = bad_counter = updates = tot_example = cum_loss = 0
    patience = 30

    display_freq = 10
    valid_freq = args.valid_freq
    batch_size = args.batch_size

    if args.model_arc == "char_cnn":
        model = vanilla_NER_CRF_model(args, ner_data_loader)
    elif args.model_arc == "char_birnn":
        model = BiRNN_CRF_model(args, ner_data_loader)
    else:
        raise NotImplementedError

    # model = debug_vanilla_NER_CRF_model(args, ner_data_loader)
    inital_lr = args.init_lr
    trainer = dy.MomentumSGDTrainer(model.model, inital_lr, 0.9)

    def _check_batch_token(batch, id_to_vocab):
        for line in batch:
            print [id_to_vocab[i] for i in line]

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print [u" ".join([id_to_vocab[c] for c in w]) for w in line]

    lr_decay = 0.05

    valid_history = []
    while epoch <= args.tot_epochs:
        for b_sents, b_char_sents, b_ner_tags, b_feats in make_bucket_batches(
                zip(sents, char_sents, tgt_tags, discrete_features), batch_size):
            dy.renew_cg()

            # _check_batch_token(b_sents, ner_data_loader.id_to_word)
            # _check_batch_token(b_ner_tags, ner_data_loader.id_to_tag)
            # _check_batch_char(b_char_sents, ner_data_loader.id_to_char)
            loss = model.cal_loss(b_sents, b_char_sents, b_ner_tags, b_feats, training=True)
            loss_val = loss.value()
            cum_loss += loss_val * len(b_sents)
            tot_example += len(b_sents)

            # cum_loss += loss_val * len(b_sents) * len(b_sents[0])
            # tot_example += len(b_sents) * len(b_sents[0])

            updates += 1
            loss.backward()
            trainer.update()

            if updates % display_freq == 0:
                # print("avg sum score = %f, avg sent score = %f" % (sum_s.value(), sent_s.value()))
                print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (epoch, updates, loss_val, cum_loss*1.0/tot_example))
            if updates % valid_freq == 0:
                if not args.isLr:
                    acc, precision, recall, f1 = evaluate(ner_data_loader, args.test_path, model)
                else:
                    acc, precision, recall, f1 = evaluate_lr(ner_data_loader, args.test_path, model)

                if len(valid_history) == 0 or f1 > max(valid_history):
                    bad_counter = 0
                    best_results = [acc, precision, recall, f1]
                else:
                    bad_counter += 1
                if bad_counter > patience:
                    print("Early stop!")
                    print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                    # test_acc, test_precision, test_recall, test_f1 = evaluate_lr()
                    exit(0)
                valid_history.append(f1)
        epoch += 1
        print("Epoch = %d, Learning Rate = %f." % (epoch, inital_lr/(1+epoch*lr_decay)))
        trainer = dy.MomentumSGDTrainer(model.model, inital_lr/(1+epoch*lr_decay))

    # #Making output ready for Darpa format
    # ## evaluation
    # if args.setEconll is not None:
    #     os.system("python ../models/Convert_Output_Darpa.py --input ../eval/pred_output.conll  --setEconll %s  --output ../eval/conv_to_darpa.conll" % (args.setEconll))
    #     #os.system("python /Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/models/Convert_to_darpa_xml.py --input /Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/eval/conv_to_darpa.conll --output /Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/eval/darpa_Ready.xml")

# add task specific trainer and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", default=5783287, type=int)

    parser.add_argument("--lang", default="english", help="the target language")
    parser.add_argument("--train_path", default="../datasets/english/eng.train.bio.conll", type=str)
    # parser.add_argument("--train_path", default="../datasets/english/debug_train.bio", type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)

    parser.add_argument("--model_arc", default="char_cnn", choices=["char_cnn", "char_birnn"], type=str)
    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=50, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--char_hidden_dim", default=25, type=int, help="char level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)

    parser.add_argument("--replace_unk_rate", default=0.0, type=float, help="uses when not all words in the test data is covered by the pretrained embedding")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--map_pretrain", default=False, action="store_true")
    parser.add_argument("--map_dim", default=100, type=int)
    parser.add_argument("--pretrain_fix", default=False, action="store_true")

    parser.add_argument("--output_dropout_rate", default=0.5, type=float, help="dropout applied to the output of birnn before crf")
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float, help="dropout applied to the input of token-level birnn")
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--init_lr", default=0.015, type=float)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--data_aug", default=False, action="store_true", help="If use data_aug, the train_path should be the combined training file")
    parser.add_argument("--aug_lang", default="english", help="the language to augment the dataset")
    parser.add_argument("--aug_lang_train_path", default=None, type=str)
    parser.add_argument("--tgt_lang_train_path", default="../datasets/english/eng.train.bio.conll", type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--pretrain_finetune", default="False", action="store_true")

    parser.add_argument("--use_discrete_features", default=False, action="store_true")
    parser.add_argument("--feature_dim", type=int, default=30)
    parser.add_argument("--isLr", default=False, action="store_true")
    parser.add_argument("--setEconll", type=str, default=None)
    args = parser.parse_args()
    main(args)
