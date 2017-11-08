#!/usr/bin/env bash
MODEL_NAME=$1
python ../main.py \
    --dynet-seed 5783287 \
    --word_emb_dim 64 \
    --batch_size 10 \
    --train_path ../datasets/german/deu.train.utf8.conll \
    --dev_path ../datasets/german/deu.testa.utf8.conll \
    --test_path ../datasets/german/deu.testb.utf8.conll \
    --pretrain_emb_path ../datasets/embs/sskip/ger_emb.txt \
    --emb_dropout_rate 0.0 \
    --output_dropout_rate 0.5 \
    --init_lr 0.01 \
    --model_arc char_birnn \
    --tag_emb_dim 100 \
    --hidden_dim 100 \
    --char_emb_dim 30\
    --char_hidden_dim 25 \
    --lang german \
    --replace_unk_rate 0.5 \
    --valid_freq 1300 2>&1 | tee ${MODEL_NAME}
