#!/bin/bash
TARGET_STRUCTURE_PATH=$1
TARGET_SEQUENCE="$(cat $TARGET_STRUCTURE_PATH)"

source thirdparty/miniconda/miniconda/bin/activate mcts
/usr/bin/time -f"%U" python thirdparty/mcts/src/MCTS-RNA.py -s $TARGET_SEQUENCE -GC 0.8 -d 0.01 -pk 0 
