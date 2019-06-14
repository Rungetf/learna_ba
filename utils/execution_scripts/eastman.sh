#!/bin/bash
TARGET_STRUCTURE_PATH=$1
TARGET_SEQUENCE="$(cat $TARGET_STRUCTURE_PATH)"

source thirdparty/miniconda/miniconda/bin/activate eastman
cd thirdparty/eastman  # Model files need to be loaded with a relative path
/usr/bin/time -f"%U" python src/solve_one_puzzle.py $TARGET_SEQUENCE
