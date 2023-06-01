for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/zero_shot/coco_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --remove_mean \
        --train \
        --test \
        --random_seed $seed \
        --val_eval \
        --cross_modal_val
done