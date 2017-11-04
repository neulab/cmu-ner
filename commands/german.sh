#!/usr/bin/env bash
python ../main.py \
    --dynet-seed 5783287 \
    --word_emb_dim 64 \
    --batch_size 10 \
    --train_path ../datasets/german/deu.train.utf8.conll \
    --dev_path ../datasets/german/deu.testa.utf8.conll \
    --test_path ../datasets/german/deu.testb.utf8.conll \
    --pretrain_emb_path ../datasets/ger_emb.txt \
    --output_dropout_rate 0.5 \
    --init_lr 0.01 \
    --model_arc char_birnn \
    --emb_dropout_rate 0.5 \
    --output_dropout_rate 0.0 \
    --lang german \
    --valid_freq 1300