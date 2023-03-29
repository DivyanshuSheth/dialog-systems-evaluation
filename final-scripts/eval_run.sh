#! /bin/bash

export CUDA_VISIBLE_DEVICES=7

python3 eval_test_datasets.py \
	--model_checkpoint "/nlsasfs/home/ttbhashini/arroy/bishal/dialog-systems-evaluation/final-scripts/saved-models/flan-t5-large-5e-5-lr-pc_usr-test-NjY1MzQ4/"
