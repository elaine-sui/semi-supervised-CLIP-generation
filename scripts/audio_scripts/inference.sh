python3 run.py \
    --config configs/new_datasets/audio_stage1_unimodal_text_reconstruction.yaml \
    --normalize_prefix \
    --test \
    --test_split train \
    --checkpoint /pasteur/u/esui/data/audio/ckpt/audio_stage1_unimodal_text_reconstruction_train_normed_mlp_seed_1234/2023_05_23_22_32_47/last.ckpt
    # --checkpoint /pasteur/u/esui/data/video/ckpt/video_stage1_unimodal_text_reconstruction_train_normed_mlp_seed_1234/2023_05_23_18_17_59/epoch=9-step=400.ckpt