__author__ = 'chuntingzhou'
import argparse
from dataloaders.data_loader import *
from models.model_builder import *
import os


def evaluate(data_loader, path, model):
    sents, char_sents, tgt_tags, discrete_features = data_loader.get_data_set(path, args.lang)

    # tot_acc = 0.0
    predictions = []
    gold_standards = []
    i = 0
    for sent, char_sent, tgt_tag, discrete_feature in zip(sents, char_sents, tgt_tags, discrete_features):
        sent, char_sent, discrete_feature = [sent], [char_sent], [discrete_feature]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature)

        assert len(best_path) == len(tgt_tag)
        # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
        # tot_acc += acc
        predictions.append(best_path)
        gold_standards.append(tgt_tag)

        i += 1
        if i % 1000 == 0:
            print "Testing processed %d lines " % i

    with open("../eval/pred_output.txt", "w") as fout:
        for pred, gold in zip(predictions, gold_standards):
            for p, g in zip(pred, gold):
                fout.write("XXX " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    os.system("../eval/conlleval.v2 < ../eval/pred_output.txt > eval_score.txt")

    with open("eval_score.txt", "r") as fin:
        lid = 0
        for line in fin:
            if lid == 1:
                fields = line.split(";")
                acc = float(fields[0].split(":")[1].strip()[:-1])
                precision = float(fields[1].split(":")[1].strip()[:-1])
                recall = float(fields[2].split(":")[1].strip()[:-1])
                f1 = float(fields[3].split(":")[1].strip())
            lid += 1

    output = open("eval_score.txt", "r").read().strip()
    print output
    os.system("rm eval_score.txt")

    return acc, precision, recall, f1


def main(args):
    ner_data_loader = NER_DataLoader(args)

    print ner_data_loader.id_to_tag

    if not args.data_aug:
        sents, char_sents, tgt_tags, discrete_features = ner_data_loader.get_data_set(args.train_path, args.lang)
    else:
        sents_tgt, char_sents_tgt, tags_tgt, dfs_tgt = ner_data_loader.get_data_set(args.tgt_lang_train_path, args.lang)
        sents_aug, char_sents_aug, tags_aug, dfs_aug = ner_data_loader.get_data_set(args.aug_lang_train_path, args.aug_lang)
        sents, char_sents, tgt_tags, discrete_features = sents_tgt+sents_aug, char_sents_tgt+char_sents_aug, tags_tgt+tags_aug, dfs_tgt+dfs_aug

    epoch = bad_counter = updates = cum_acc = tot_example = cum_loss = 0
    patience = 20

    display_freq = 10
    valid_freq = args.valid_freq
    batch_size = args.batch_size

    model = vanilla_NER_CRF_model(args, ner_data_loader)
    trainer = dy.MomentumSGDTrainer(model.model, 0.015, 0.9)

    def _check_batch_token(batch, id_to_vocab):
        for line in batch:
            print [id_to_vocab[i] for i in line]

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print [u" ".join([id_to_vocab[c] for c in w]) for w in line]

    valid_history = []
    while epoch <= args.tot_epochs:
        epoch += 1
        for b_sents, b_char_sents, b_ner_tags, b_feats in make_bucket_batches(
                zip(sents, char_sents, tgt_tags, discrete_features), batch_size):
            dy.renew_cg()

            # _check_batch_token(b_sents, ner_data_loader.id_to_word)
            # _check_batch_token(b_ner_tags, ner_data_loader.id_to_tag)
            # _check_batch_char(b_char_sents, ner_data_loader.id_to_char)
            loss = model.cal_loss(b_sents, b_char_sents, b_ner_tags, b_feats)
            loss_val = loss.value()
            cum_loss += loss_val * len(b_sents)
            tot_example += len(b_sents)

            updates += 1
            loss.backward()
            trainer.update()

            if updates % display_freq == 0:
                # print("avg sum score = %f, avg sent score = %f" % (sum_s.value(), sent_s.value()))
                print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (epoch, updates, loss_val, cum_loss*1.0/tot_example))
            if updates % valid_freq == 0:
                acc, precision, recall, f1 = evaluate(ner_data_loader, args.test_path, model)
                if len(valid_history) == 0 or f1 > max(valid_history):
                    bad_counter = 0
                    best_results = [acc, precision, recall, f1]
                else:
                    bad_counter += 1
                if bad_counter > patience:
                    print("Early stop!")
                    print("Best acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                    exit(0)
                valid_history.append(f1)

# add task specific trainer and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", type=int)

    parser.add_argument("--lang", default="english", help="the target language")
    parser.add_argument("--train_path", default="../datasets/english/eng.train.bio.conll", type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)

    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=50, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--layer", default=1, type=int)

    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--data_aug", default=False, action="store_true", help="If use data_aug, the train_path should be the combined training file")
    parser.add_argument("--aug_lang", default="english", help="the language to augment the dataset")
    parser.add_argument("--aug_lang_train_path", default=None, type=str)
    parser.add_argument("--tgt_lang_train_path", default="../datasets/english/eng.train.bio.conll", type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--pretrain_finetune", default="False", action="store_true")

    parser.add_argument("--use_discrete_features", default=False, action="store_true")
    parser.add_argument("--feature_dim", type=int, default=50)
    args = parser.parse_args()
    main(args)