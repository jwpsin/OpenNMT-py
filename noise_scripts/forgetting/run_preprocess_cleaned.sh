#!/bin/bash

NAME=schneider50k-cleaned/20perc
DATADIR=../../noise_data/${NAME}

src=precursors
tgt=product

onmt_preprocess -train_src $DATADIR/$src-train.txt \
                              -train_tgt $DATADIR/$tgt-train.txt \
                              -valid_src $DATADIR/$src-train.txt \
                              -valid_tgt $DATADIR/$tgt-train.txt \
                              -save_data $DATADIR/preprocess/preprocessed_onmt36 \
                              -src_seq_length 3000 -tgt_seq_length 3000 \
                              -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab