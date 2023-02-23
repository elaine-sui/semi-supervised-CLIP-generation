#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur3"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="stage2_img_cap_add_noise_during_pretrain_remove_mean"
#SBATCH --output=./log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/semi-supervised-CLIP-generation/scripts/stage2_image_captioning/train_stage2_image_captioning_remove_mean_add_noise.sh