#!/bin/bash

LOGDIR=./logs/$(date +%s)
mkdir $LOGDIR
echo Log directory $LOGDIR

for j in {0..49} # How many seeds?
do
    SEED=$RANDOM
    echo Seed is $SEED

    if [ $1 = 1 ] # Turn on k-folding
    then
        for i in {0..4} # Number of k-folds
        do
            CUDA_VISIBLE_DEVICES=$i python ./CNNs/cnnAug.py $i -S $SEED -l $LOGDIR -e 8 >$LOGDIR/$SEED-$i.log 2>&1 &
        done
        wait

    else
        for i in {0..5} # Number of available GPUs
        do
            CUDA_VISIBLE_DEVICES=$i python ./CNNs/cnnAug.py -S $SEED -e 8 -l $LOGDIR >$LOGDIR/$SEED.log 2>&1 &
        done
        wait
    fi
done
