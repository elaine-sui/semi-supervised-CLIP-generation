python3 run.py \
    --config configs/zero_shot/imagebind_coco_stage1_unimodal_text_reconstruction.yaml \
    --normalize_prefix \
    --add_gaussian_noise \
    --train \
    --test \
    --cross_modal_val \
    --noise_level 0.5 \
    --val_eval \
    --re_normalize_prefix \