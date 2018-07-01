#!/bin/bash
SEED=$RANDOM

for i in {0..4}
do
    python ./CNNs/cnnAug.py $i -s $SEED
done
