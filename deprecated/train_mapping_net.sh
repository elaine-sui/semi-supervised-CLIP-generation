python train.py \
	--only_prefix \
	--data /pasteur/u/esui/data/coco/oscar_split_ViT-B_32_train.pkl \
	--out_dir /pasteur/u/esui/data/coco/mapping_net_ckpts/ \
	--mapping_type transformer \
	--num_layers 8 \
	--prefix_length 40 \
	--prefix_length_clip 40
