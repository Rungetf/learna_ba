#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 163 \
  --conv_sizes 0 0 \
  --conv_channels 10 2 \
  --embedding_size 6 \
  --entropy_regularization 0.00443007543643681 \
  --fc_units 58 \
  --learning_rate 0.00035282057818765335 \
  --lstm_units 38 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 10.729068367266082 \
  --state_radius 17 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --restart_timeout 1800
