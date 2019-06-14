#!/bin/bash
TARGET_STRUCTURE_PATH=$1
TARGET_SEQUENCE="$(cat $TARGET_STRUCTURE_PATH)"

source thirdparty/miniconda/miniconda/bin/activate rnainverse
/usr/bin/time -f"%U" RNAinverse -R-1 < $TARGET_STRUCTURE_PATH
