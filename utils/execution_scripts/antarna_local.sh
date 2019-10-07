#!/bin/bash
TARGET_STRUCTURE_PATH=$1
TARGET_STRUCTURE="$(cut -f1 $TARGET_STRUCTURE_PATH)"
SEQUENCE_CONTRAINTS="$(cut -f2 $TARGET_STRUCTURE_PATH)"

source thirdparty/miniconda/miniconda/bin/activate antarna
/usr/bin/time -f"%U" python thirdparty/antarna/src/antaRNA_v114.py -Cstr $TARGET_STRUCTURE -Cseq $SEQUENCE_CONTRAINTS -tGC 0.5 -n 1 -noLBP -noGU -r 2147483647 -t 2147483647 -Cgcw 0.0 -ov
