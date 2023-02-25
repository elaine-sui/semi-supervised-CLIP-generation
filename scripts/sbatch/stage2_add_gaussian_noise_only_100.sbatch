#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur3"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="stage2_img_cap_add_noise_during_pretrain"
#SBATCH --output=./log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --normalize_prefix --checkpoint /pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_train+restval_normed_mlp_add_gaussian_noise_seed_1234/2023_02_20_18_48_08/last.ckpt