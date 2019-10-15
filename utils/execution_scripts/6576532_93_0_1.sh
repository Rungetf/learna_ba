#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 213 \
  --conv_sizes 3 0 \
  --conv_channels 4 10 \
  --embedding_size 6 \
  --entropy_regularization 9.842102334353838e-05 \
  --fc_units 19 \
  --learning_rate 0.00042606100083982624 \
  --lstm_units 2 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 7.2294513125370194 \
  --state_radius 30 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --predict_pairs \
  --restore_path local_models/93_0_1/ \
  --stop_learning
