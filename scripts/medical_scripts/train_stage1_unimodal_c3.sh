for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/new_datasets/medical_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --add_gaussian_noise \
        --remove_mean \
        --train \
        --test \
        --noise_level 0.1 \
        --random_seed $seed
done