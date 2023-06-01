for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/zero_shot/audio_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --add_gaussian_noise \
        --remove_mean \
        --train \
        --test \
        --random_seed $seed \
        --val_eval \
        --cross_modal_val \
        --noise_level 0.1
done