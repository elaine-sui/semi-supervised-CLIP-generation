CHECKPOINT=/pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_train+restval_normed_mlp_remove_mean_add_gaussian_noise_seed_1234/2023_02_24_09_25_03/last.ckpt

for seed in 1234 5678 910
do
#     # 1% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.01 --random_seed $seed --normalize_prefix --remove_mean --checkpoint $CHECKPOINT

#     # 5% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.05 --random_seed $seed --normalize_prefix --remove_mean --checkpoint $CHECKPOINT

#     # 10% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.1 --random_seed $seed --normalize_prefix --remove_mean --checkpoint $CHECKPOINT

    # 25% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.25 --random_seed $seed --normalize_prefix --remove_mean --checkpoint $CHECKPOINT
done

python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --random_seed $seed --normalize_prefix --remove_mean --checkpoint $CHECKPOINT