for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/new_datasets/video_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --add_gaussian_noise \
        --train \
        --test \
        --noise_level 0.0001 \
        --random_seed $seed
done