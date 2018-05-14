#!/usr/bin/env sh

inpt_folder=$1
cam=$2
interval=$3

for x in $(seq 0 $interval 176); do
    for y in $(seq 0 $interval 71); do
        ./scripts/preprocess.py --data_path $inpt_folder --cam $cam --yx $y,$x
    done
done
