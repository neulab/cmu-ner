#!/usr/bin/env bash
python ../main.py \
    --dynet-seed 3278657 \
    --word_emb_dim 100 \
    --batch_size 10 \
    --pretrain_emb_path ../datasets/english/glove.6B/glove.6B.100d.txt\
    --lang eng
