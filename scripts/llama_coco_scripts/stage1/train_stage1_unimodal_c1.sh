CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --config configs/zero_shot/llama_coco_stage1_unimodal_text_reconstruction.yaml \
    --normalize_prefix \
    --train \
    --test \
    --cross_modal_val \
    --val_eval \
    --re_normalize_prefix \
    --debug