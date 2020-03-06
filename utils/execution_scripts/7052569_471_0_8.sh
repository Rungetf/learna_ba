#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 118 \
  --conv_channels 2 25 \
  --conv_sizes 0 5 \
  --data_type "random-sort" \
  --embedding_size 2 \
  --entropy_regularization 7.990216536104802e-05 \
  --fc_units 49 \
  --learning_rate 0.000854036796946988 \
  --lstm_units 1 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 10.892910865085273 \
  --reward_function "structure_only" \
  --state_radius 24 \
  --state_representation "sequence_progress" \
  --local_design \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/7052569/471_0_8/ \
  --stop_learning
