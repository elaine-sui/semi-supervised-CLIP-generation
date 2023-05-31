for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/new_datasets/audio_stage1_unimodal_text_reconstruction_val_eval.yaml \
        --normalize_prefix \
        --train \
        --test \
        --random_seed $seed \
        --val_eval \
        --cross_modal_val
done