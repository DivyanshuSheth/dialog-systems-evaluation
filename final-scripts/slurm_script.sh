#! /bin/bash
## --- SLURM JOB SUBMISSION SCRIPT --- ##
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=1-01:00:00
##SBATCH --error=/nlsasfs/home/ttbhashini/arroy/bishal/dialog-systems-evaluation/final-scripts/logs/job_%j.%3t.err
#SBATCH --output=/nlsasfs/home/ttbhashini/arroy/bishal/dialog-systems-evaluation/final-scripts/logs/job_%j.%3t.out
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

#################conda environment path ################################
source /nlsasfs/home/ttbhashini/arroy/anaconda3/bin/activate
gpustat

# Activate 
conda activate py38-bishal

# Workdir
cd /nlsasfs/home/ttbhashini/arroy/bishal/dialog-systems-evaluation/final-scripts/

# wandb
export WANDB_API_KEY=e59936a0af66d4bd898d799e16f2ae4fcfa23ead
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Remember the diff
git diff

# Run script
python3 -u process-data-and-train.py \
	--train_batch_size 4 \
	--test_datasets "pc_usr,tc_usr" \
	--save_steps 20000 \
	--eval_steps 20000 \
	--logging_steps 200 \
	--num_epochs 10 \
	--model_checkpoint google/flan-t5-base
