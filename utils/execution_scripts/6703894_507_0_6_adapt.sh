#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --mutation_threshold 5 \
  --batch_size 180 \
  --conv_sizes 0 7 \
  --conv_channels 17 10 \
  --embedding_size 2 \
  --entropy_regularization 0.0005479661372337776 \
  --fc_units 51 \
  --learning_rate 0.0005767507411238691 \
  --lstm_units 13 \
  --num_fc_layers 2 \
  --num_lstm_layers 0 \
  --reward_exponent 11.52071779218975 \
  --state_radius 19 \
  --reward_function "structure_only" \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --local_design \
  --predict_pairs \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/6703894/507_0_6/ 
