python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.1 --random_seed 5678 --normalize_prefix

python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.1 --random_seed 5678 --remove_modality_gap --normalize_prefix

python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.25 --random_seed 5678 --normalize_prefix

python3 run.py --config configs/coco_stage2_image_captioning_from_unimodal_restval.yaml --train --test --sample_frac 0.25 --random_seed 5678 --remove_modality_gap --normalize_prefix