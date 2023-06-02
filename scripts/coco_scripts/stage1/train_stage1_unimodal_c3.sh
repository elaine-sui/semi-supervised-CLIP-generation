python3 run.py \
    --config configs/zero_shot/coco_stage1_unimodal_text_reconstruction.yaml \
    --normalize_prefix \
    --add_gaussian_noise \
    --remove_mean \
    --train \
    --test \
    --val_eval \
    --cross_modal_val \
    --re_normalize_prefix