# python3 run.py --config configs/coco_stage1_unimodal_text_reconstruction.yaml --train --test

python3 run.py \
    --config configs/coco_stage1_unimodal_text_reconstruction.yaml \
    --test \
    --debug \
    --checkpoint '/pasteur/u/esui/data/coco/ckpt/coco_stage1_unimodal_text_reconstruction_seed_1234/2023_02_07_22_49_16/last.ckpt'