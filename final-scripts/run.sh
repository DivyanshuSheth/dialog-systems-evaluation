#!/bin/bash

echo "Running process-data-and-train.py"
python3 process-data-and-train.py --test_datasets "pc_usr,tc_usr" --save_steps 1000 --eval_steps 1000 --logging_steps 100