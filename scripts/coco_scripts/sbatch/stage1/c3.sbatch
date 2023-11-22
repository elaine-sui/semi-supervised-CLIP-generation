#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name="coco_c3"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash ./scripts/coco_scripts/stage1/train_stage1_unimodal_c3.sh 1234
# bash ./scripts/coco_scripts/stage1/train_stage1_unimodal_c3.sh 5678
# bash ./scripts/coco_scripts/stage1/train_stage1_unimodal_c3.sh 910