#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 248 \
  --conv_sizes 11 9 \
  --conv_channels 18 15 \
  --embedding_size 5 \
  --entropy_regularization 5.436517475731531e-05 \
  --fc_units 60 \
  --learning_rate 0.0005824327013981257 \
  --lstm_units 2 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 4.9241093473795505 \
  --state_radius 22 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --predict_pairs \
  --restore_path local_models/482_0_0/ \
  --stop_learning
