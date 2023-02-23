CHECKPOINT=/pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_train+restval_normed_mlp_add_gaussian_noise_seed_1234/2023_02_20_18_48_08/last.ckpt

for seed in 1234 5678 910
do
    # 1% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.01 --random_seed $seed --normalize_prefix --checkpoint $CHECKPOINT

    # 5% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.05 --random_seed $seed --normalize_prefix --checkpoint $CHECKPOINT

    # 10% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.1 --random_seed $seed --normalize_prefix --checkpoint $CHECKPOINT

    # 25% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.25 --random_seed $seed --normalize_prefix --checkpoint $CHECKPOINT
done