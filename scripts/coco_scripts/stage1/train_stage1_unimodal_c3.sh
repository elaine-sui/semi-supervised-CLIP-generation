python3 run.py \
    --config configs/zero_shot/coco_stage1_og.yaml \
    --normalize_prefix \
    --add_gaussian_noise \
    --remove_mean \
    --train \
    --test \
    --cross_modal_val \
    --val_eval \
    --re_normalize_prefix \
    --random_seed $1

# coco_stage1_unimodal_text_reconstruction.yaml \