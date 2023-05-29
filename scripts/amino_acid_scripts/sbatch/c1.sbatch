#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclude="pasteur1"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="c1"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/semi-supervised-CLIP-generation/scripts/amino_acid_scripts/train_stage1_unimodal_c1.sh