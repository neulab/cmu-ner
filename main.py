import argparse
from dataloaders.data_loader import *
from models.model_builder import *

def main(args):
    ner_data_loader = NER_DataLoader(args.train_path)
    sents, char_sents, tgt_tags, discrete_features = ner_data_loader.get_data_set()
    bad_counter = updates = cum_acc = tot_example = 0
    patience = 20

    display_freq = 100
    valid_freq = args.valid_freq
    batch_size = args.batch_size

    model = vanilla_NER_CRF_model(args, ner_data_loader)
    trainer = dy.MomentumSGDTrainer(model, 0.015, 0.9)


    for b_sents, b_char_sents, b_ner_tags, b_feats in make_bucket_batches(zip(sents, char_sents, tgt_tags, discrete_features), batch_size):
        loss = model.forward(b_sents, b_char_sents, b_ner_tags, b_feats)

        loss.backward()
        trainer.update()


# add task specific trainer and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", type=int)

    parser.add_argument("--train_path", default="./datasets/NER/english/eng.train.bio.conll", type=str)
    parser.add_argument("--dev_path", default="./datasets/NER/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="./datasets/NER/english/eng.test.bio.conll", type=str)

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

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)
    parser.add_argument("--data_aug", default=False, action="store_true")
    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--use_discrete_features", default=False, action="store_true")
    parser.add_argument("--feature_dim", type=int, default=50)
    args = parser.parse_args()
    main(args)