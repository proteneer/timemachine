#!/bin/bash
for gidx in {0..10}
do
    # echo $port
    python -u parallel/worker.py --gpu_idx $gidx --port $(($gidx+5000)) > log_$gidx.txt &
done
