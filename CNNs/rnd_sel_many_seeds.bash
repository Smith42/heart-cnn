#!/bin/bash
for j in {0..49} # How many seeds?
do
    for i in {0..5}
    do
        SEED=$RANDOM
        CUDA_VISIBLE_DEVICES=$i python ./CNNs/cnnAug.py -s $SEED -e 8 >./temp/$SEED.log 2>&1 &
    done

    wait
done
