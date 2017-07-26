#!/bin/bash

for i in {0..2}
do
    echo $i
    python ./3D-2ch-fakedata/cnn.py $i
done
