#!/bin/bash
for port in {0..5}
do
    # echo $port
    python -u training/worker.py --gpu_idx 0 --port $(($port+50000)) --registry localhost:582734 > log_$port.txt &
done
