#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="coco_c2_noise_01"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash ./scripts/coco_scripts/stage1/train_stage1_unimodal_c2_noise_01.sh