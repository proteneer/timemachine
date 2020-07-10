#!/bin/bash
for port in {0..9}
do
    # echo $port
    python -u training/worker.py --gpu_idx $port --port $(($port+50000)) > log_$port.txt &
done
