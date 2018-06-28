#!/bin/bash

for i in {0..4}
do
    echo $i
    python ./CNNs/cnnAug.py $i
done
