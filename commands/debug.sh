#!/usr/bin/env bash
python ../main.py \
    --dynet-seed 3278657 \
    --word_emb_dim 100 \
    --batch_size 10 \
    --model_name "eng" \
    --lang eng \
    --valid_freq 1300

#   --pretrain_emb_path ../new_datasets/embs/glove.6B.100d.txt\
