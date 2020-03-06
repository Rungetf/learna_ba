#!/bin/bash
TARGET_STRUCTURE_PATH=$1

source thirdparty/miniconda/miniconda/bin/activate learna
/usr/bin/time -f"%U" python -m src.learna.design_rna \
  --target_structure_path $TARGET_STRUCTURE_PATH \
  --mutation_threshold 5 \
  --batch_size 76 \
  --conv_channels 10 6 \
  --conv_sizes 9 9 \
  --data_type "random-sort" \
  --embedding_size 5 \
  --entropy_regularization 0.0008294550760143675 \
  --fc_units 32 \
  --learning_rate 0.00018650406453277055 \
  --lstm_units 8 \
  --num_fc_layers 2 \
  --num_lstm_layers 0 \
  --predict_pairs \
  --reward_exponent 11.364077406127976 \
  --reward_function "structure_only" \
  --state_radius 22 \
  --state_representation "sequence_progress" \
  --local_design \
  --restore_path /work/ws/nemo/fr_ds371-learna-0/results/bohb/6782534/414_0_5/ \
	--restart_timeout 1800 \
  --stop_learning
