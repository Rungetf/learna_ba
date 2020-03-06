#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 101 \
  --conv_channels 4 7 \
  --conv_sizes 0 3 \
  --data_type "random" \
  --embedding_size 2 \
  --entropy_regularization 0.00452019965111065 \
  --fc_units 21 \
  --learning_rate 0.000975550712164158 \
  --lstm_units 10 \
  --num_fc_layers 1 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 11.7204878655058 \
  --reward_function "structure_only" \
  --state_radius 15 \
  --state_representation "n-gram" \
  --local_design \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/6782535/312_0_8/ \
	--restart_timeout 1800
