#!/bin/bash

for i in {0..4}
do
    echo $i
    python ./finetuning/cnnFinetune.py $i
done
