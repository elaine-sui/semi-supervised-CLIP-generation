for seed in 1234 5678 910
do
#     # 1% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval_indirect.yaml --train --test --sample_frac 0.01 --random_seed $seed --remove_mean --normalize_prefix

#     # 5% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval_indirect.yaml --train --test --sample_frac 0.05 --random_seed $seed --remove_mean --normalize_prefix

#     # 10% of pre-training data size
#     python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval_indirect.yaml --train --test --sample_frac 0.1 --random_seed $seed --remove_mean --normalize_prefix

    # 25% of pre-training data size
    python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval_indirect.yaml --train --test --sample_frac 0.25 --random_seed $seed --remove_mean --normalize_prefix
done