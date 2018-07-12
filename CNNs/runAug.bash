#!/bin/bash
SEED=$RANDOM

for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=$i python ./CNNs/cnnAug.py $i -s $SEED >$SEED-$i.log 2>&1 &
done
