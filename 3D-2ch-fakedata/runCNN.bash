#!/bin/bash

for i in {0..2}
do
    echo $i
    python ./cnn.py $i
done
