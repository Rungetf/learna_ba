#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 148 \
  --conv_channels 2 2 \
  --conv_sizes 0 0 \
  --embedding_size 5 \
  --entropy_regularization 0.0057625704783584265 \
  --fc_units 50 \
  --learning_rate 0.0006850571396934639 \
  --lstm_units 2 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 9.24060577555553 \
  --state_radius 23 \
  --reward_function "structure_only" \
  --state_representation "sequence_progress" \
  --local_design \
  --restart_timeout 1800
