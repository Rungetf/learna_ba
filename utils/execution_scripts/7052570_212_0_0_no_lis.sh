#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 0 \
  --batch_size 95 \
  --conv_channels 23 2 \
  --conv_sizes 7 0 \
  --data_type "random" \
  --embedding_size 1 \
  --entropy_regularization 0.002846838967807086 \
  --fc_units 61 \
  --learning_rate 0.00035728046577710495 \
  --lstm_units 3 \
  --num_fc_layers 2 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 10.78740103642566 \
  --reward_function "structure_only" \
  --state_radius 22 \
  --state_representation "sequence_progress" \
  --local_design \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/7052570/212_0_0/ 
