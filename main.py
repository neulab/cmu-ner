__author__ = 'chuntingzhou'
import argparse
from dataloaders.data_loader import *
from models.model_builder import *

def evaluate(data_loader, path, model):
    sents, char_sents, tgt_tags, discrete_features = data_loader.get_data_set(path)

    tot_acc = 0.0
    for sent, char_sent, tgt_tag, discrete_feature in zip(sents, char_sents, tgt_tags, discrete_features):
        sent, char_sent, discrete_feature = [sent], [char_sent], [discrete_feature]
        best_score, best_path = model.eval(sent, char_sent, discrete_feature)

        assert len(best_path) == len(tgt_tag)
        acc = model.crf_decoder.cal_accuracy(best_path, tgt_tags)
        tot_acc += acc
    return tot_acc / len(sents)

def main(args):
    ner_data_loader = NER_DataLoader(args)
    print "hhhh"
    sents, char_sents, tgt_tags, discrete_features = ner_data_loader.get_data_set(args.train_path)
    print "data load ends"
    epoch = bad_counter = updates = cum_acc = tot_example = 0
    patience = 20

    display_freq = 10
    valid_freq = args.valid_freq
    batch_size = args.batch_size

    model = vanilla_NER_CRF_model(args, ner_data_loader)
    trainer = dy.MomentumSGDTrainer(model, 0.015, 0.9)

    valid_history = []
    while epoch <= args.tot_epochs:
        epoch += 1
        for b_sents, b_char_sents, b_ner_tags, b_feats in make_bucket_batches(
                zip(sents, char_sents, tgt_tags, discrete_features), batch_size):
            dy.renew_cg()
            print "hhhhhhh"
            loss = model.cal_loss(b_sents, b_char_sents, b_ner_tags, b_feats)
            loss_val = loss.value()

            updates += 1
            loss.backward()
            trainer.update()

            if display_freq % updates == 0:
                print("Epoch = %d, Updates = %d, CRF Loss=%f." % (epoch, updates, loss_val))
            if updates % valid_freq == 0:
                pass


# add task specific trainer and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", type=int)

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
    parser.add_argument("--tot_epochs", default=100)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)
    parser.add_argument("--data_aug", default=False, action="store_true")
    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--pretrain_finetune", default="False", action="store_true")
    parser.add_argument("--use_discrete_features", default=False, action="store_true")
    parser.add_argument("--feature_dim", type=int, default=50)
    args = parser.parse_args()
    main(args)