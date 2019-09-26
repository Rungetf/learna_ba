#!/bin/bash
TARGET_STRUCTURE_PATH=$1
TARGET_SEQUENCE="$(cat $TARGET_STRUCTURE_PATH)"

source thirdparty/miniconda/miniconda/bin/activate antarna
/usr/bin/time -f"%U" python thirdparty/antarna/src/antaRNA_v114.py -Cstr $TARGET_SEQUENCE -tGC 0.1 -n 1 -noLBP -r 2147483647 -t 2147483647 -ov -tGCvar 0.01
