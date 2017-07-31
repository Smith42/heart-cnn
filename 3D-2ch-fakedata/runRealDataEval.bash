#!/bin/bash

for i in {0..4}
do
    echo $i
    python ./3D-2ch-fakedata/realDataComparison.py $i
done
