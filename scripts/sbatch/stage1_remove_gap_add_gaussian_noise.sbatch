#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur3"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="img_cap_remove_gap_add_noise"
#SBATCH --output=./log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/semi-supervised-CLIP-generation/scripts/train_stage1_unimodal_text_recon_remove_gap_add_noise.sh