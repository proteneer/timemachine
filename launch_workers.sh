#!/bin/bash
for port in {0..1}
do
    # echo $port
    python -u training/worker.py --gpu_idx 0 --port $(($port+5000)) > log_$port.txt &
done
