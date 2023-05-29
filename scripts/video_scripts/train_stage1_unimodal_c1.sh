for seed in 5678 910; do
    python3 run.py \
        --config configs/new_datasets/video_stage1_unimodal_text_reconstruction.yaml \
        --normalize_prefix \
        --train \
        --test \
        --random_seed $seed
done