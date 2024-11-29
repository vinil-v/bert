#!/bin/bash
#SBATCH --job-name=bert_training
#SBATCH --nodes=2                        # Number of nodes
#SBATCH --ntasks-per-node=8              # Number of tasks (1 per GPU)
#SBATCH --gres=gpu:8                     # GPUs per node
#SBATCH --time=04:00:00                  # Adjust as needed
#SBATCH --output=bert_train_%j.log       # Output log file

# Optional: Set Slurm environment manually (if needed)
source /etc/profile.d/slurm.sh    # Load Slurm environment (optional, depending on setup)

# Set NCCL and PyTorch distributed environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0            # Use your IB interface (check with ifconfig)
export OMP_NUM_THREADS=8                 # Adjust based on CPU cores

# Launch Pyxis container with distributed training
srun --container-image pytorch_train.sqsh \
     --container-name=bert_train \
     --export=ALL \
     bash -c '
         python /workspace/bert/run_pretraining.py \
         --model_type bert \
         --model_name_or_path bert-base-uncased \
         --do_train \
         --train_file /path/to/training_data.txt \
         --output_dir /path/to/output \
         --per_device_train_batch_size 16 \
         --num_train_epochs 3 \
         --distributed_world_size 16   # Total GPUs across nodes (2 nodes x 8 GPUs)
     '