for seed in 1234 5678 910; do
    python3 run.py \
        --config configs/new_datasets/audio_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --remove_mean \
        --train \
        --test \
        --random_seed $seed
done