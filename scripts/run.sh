#!/bin/bash

#-------------------------------------------------------------------------
#SBATCH -G a100:1               # number of GPUs
#SBATCH -t 0-05:00:00          # time in d-hh:mm:ss
#SBATCH -p general                 # partition
#SBATCH -q public
#SBATCH --mem=80G
#SBATCH -o ./jobs/slurm.%x_%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e ./jobs/slurm.%x_%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL   # Send an e-mail when a job stops, or fails
#SBATCH --mail-user=%u@asu.edu # Mail-to address
#SBATCH --export=NONE          # Purge the job-submitting shell environment
#SBATCH --job-name=hopper-shac-bundle-sigma5 # Job name
#-------------------------------------------------------------------------

module load mamba
module load cuda-12.6.1-gcc-12.1.0
source activate diffrl

# start TensorBoard in the background, redirect its output to a log file
#tensorboard --logdir outputs/ --port 6006 --bind_all \
#    > outputs/tensorboard.log 2>&1 &

python train.py alg=shac env=hopper env.config.dr=true env.shac.max_epochs=1000 env.config.bundle=true env.config.num_samples=20 env.config.sigma=5
