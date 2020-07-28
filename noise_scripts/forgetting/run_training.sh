#!/bin/bash

NAME=schneider50k
DATADIR=../../data/${NAME}

EXPERIMENT_NAME='baseline'

SEED=42

onmt_train -data $DATADIR/preprocess/preprocessed_onmt36  \
                -save_model  ${DATADIR}/training/${NAME}_${EXPERIMENT_NAME}_r${SEED} \
                -seed $SEED -save_checkpoint_steps 140 -keep_checkpoint 50 \
                -train_steps 5460 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                -batch_size 6144 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                -learning_rate 2 -label_smoothing 0.0 -report_every 100  -valid_batch_size 8 \
                -layers 4 -rnn_size  384 -word_vec_size 384 -encoder_type transformer -decoder_type transformer \
                -dropout 0.1 -position_encoding -share_embeddings \
                -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                -heads 8 -transformer_ff 2048 # -gpu_ranks 0 # for gpu usage