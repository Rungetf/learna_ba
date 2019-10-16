#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 225 \
  --conv_sizes 9 7 \
  --conv_channels 8 7 \
  --embedding_size 6 \
  --entropy_regularization 2.522990252852851e-07 \
  --fc_units 51 \
  --learning_rate 0.0007265821198520795 \
  --lstm_units 1 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --reward_exponent 8.913246237038848 \
  --state_radius 26 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --predict_pairs \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/6663703/156_0_7 
