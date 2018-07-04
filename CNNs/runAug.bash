#!/bin/bash
SEED=$RANDOM

for i in {0..4}
do
    python ./CNNs/get_folds.py $i -s $SEED
    python ./CNNs/cnnAug.py $i -s $SEED
done
