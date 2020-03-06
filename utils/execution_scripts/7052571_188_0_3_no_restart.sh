#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 196 \
  --conv_channels 3 4 \
  --conv_sizes 0 0 \
  --embedding_size 7 \
  --entropy_regularization 0.000528063083283936 \
  --fc_units 26 \
  --learning_rate 0.000384121783602744 \
  --lstm_units 8 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 11.454339469173966 \
  --state_radius 0 \
  --reward_function "structure_only" \
  --state_representation "n-gram" \
  --local_design
