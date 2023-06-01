echo coco_stage1_unimodal_text_reconstruction_seed_1234
python3 run.py \
    --config configs/clip_prefix_cap_refactored_test_pretrained_only.yaml \
    --test \
    --checkpoint '/pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_seed_1234/2023_02_07_22_49_16/last.ckpt'

echo coco_stage1_unimodal_text_reconstruction_train
python3 run.py \
    --config configs/clip_prefix_cap_refactored_test_pretrained_only.yaml \
    --test \
    --checkpoint '/pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_train+restval_seed_1234/2023_02_07_23_13_00/last.ckpt'

# echo coco_single_stage_img_cap_and_text_recon_seed_1234
# python3 run.py \
#     --config configs/clip_prefix_cap_refactored_test_pretrained_only.yaml \
#     --test \
#     --checkpoint '/pasteur/u/esui/data/coco/ckpt/coco_single_stage_img_cap_and_text_recon_seed_1234/2023_02_07_22_49_19/last.ckpt'
    
# echo coco_stage2_unimodal_text_reconstruction_seed_1234
# python3 run.py \
#     --config configs/clip_prefix_cap_refactored_test_pretrained_only.yaml \
#     --test \
#     --checkpoint '/pasteur/u/esui/data/coco/ckpt/coco_stage2_unimodal_text_reconstruction_seed_1234/2023_02_07_22_49_12/last.ckpt'


# python3 run.py --config configs/clip_prefix_cap_refactored_test_pretrained_only.yaml --test --checkpoint