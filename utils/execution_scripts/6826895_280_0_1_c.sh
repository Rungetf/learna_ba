#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 44 \
  --conv_channels 3 6 \
  --conv_sizes 0 7 \
  --embedding_size 6 \
  --entropy_regularization 0.0019828416120483724 \
  --fc_units 64 \
  --learning_rate 0.0005803515427643226 \
  --lstm_units 39 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 7.207490578070082 \
  --state_radius 20 \
  --reward_function "structure_only" \
  --local_design \
  --predict_pairs \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/6826895/280_0_1/ \
  --state_representation "n-gram" \
  --sequence_constraints \
  --data_type "random"
