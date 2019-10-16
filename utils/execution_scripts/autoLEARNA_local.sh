#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 75 \
  --conv_sizes 0 0 \
  --conv_channels 8 23 \
  --embedding_size 7 \
  --entropy_regularization 0.0062538752481115885 \
  --fc_units 47 \
  --learning_rate 0.0007629922155340233 \
  --lstm_units 10 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 7.077835098205039 \
  --state_radius 22 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --restart_timeout 1800
