import torch
import pickle
import argparse

import os
import sys
sys.path.append(os.getcwd())
from src.enums import DatasetType

TEXT_VIDEOCLIP_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_videoclip_3k_embed_mean.pkl'
VIDEO_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_video_3k_embed_mean.pkl'
TEXT_CONVIRT_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_convirt_10k_embed_mean.pkl'
TEXT_MEDCLIP_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_medclip_10k_embed_mean.pkl'
MED_IMAGES_MEDCLIP_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_medclip_med_image_10k_embed_mean.pkl'
TEXT_MEDCLIP_NO_AUG_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_medclip_no_aug_10k_embed_mean.pkl'
MED_IMAGES_MEDCLIP_NO_AUG_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_medclip_no_aug_med_image_10k_embed_mean.pkl'
MED_IMAGES_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_med_image_10k_embed_mean.pkl'
TEXT_CLASP_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_clasp_embed_mean.pkl'
AMINO_ACID_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_amino_acid_embed_mean.pkl'
TEXT_AUDIO_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_imagebind_embed_mean.pkl'
AUDIO_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_audio_imagebind_embed_mean.pkl'

TEXT_IMAGEBIND_VIDEO_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_imagebind_video_embed_mean.pkl'
VIDEO_IMAGEBIND_EMBED_MEAN = '/pasteur/u/esui/data/c3/normalized_text_video_imagebind_embed_mean.pkl'

def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        data_path = '/pasteur/u/esui/data/c3/data_video_msrvtt_imagebind_train.pkl'
        text_mean_path = TEXT_IMAGEBIND_VIDEO_EMBED_MEAN
        x_mean_path = VIDEO_IMAGEBIND_EMBED_MEAN
    elif dataset_type == DatasetType.Audio:
        data_path = '/pasteur/u/esui/data/c3/data_audio_clotho_imagebind_train.pkl'
        text_mean_path = TEXT_AUDIO_EMBED_MEAN
        x_mean_path = AUDIO_EMBED_MEAN
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")
    
    return data_path, text_mean_path, x_mean_path

def main(dataset_type):
    data_path, text_mean_path, x_mean_path = get_data_paths(dataset_type)
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # {input_id: {"x": ..., "embed": ...}}
    inputs = all_data["inputs"]
    # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
    captions = all_data["captions"]
    
    text_mean = torch.zeros(1, 1024)
    input_mean = torch.zeros(1, 1024)
    
    # Compute text mean
    for cap_id in captions:
        cap_embed = captions[cap_id]["embed"]
        text_mean += cap_embed / cap_embed.norm()
    
    text_mean = text_mean / len(captions)
    
    # Compute image mean
    for input_id in inputs:
        input_embed = inputs[input_id]["embed"]
        input_mean += input_embed / input_embed.norm()
    
    input_mean = input_mean / len(inputs)
        
    with open(text_mean_path, 'wb') as f:
        pickle.dump(text_mean, f)
    
    with open(x_mean_path, 'wb') as f:
        pickle.dump(input_mean, f)
    
    print(f"Saved text_embed_mean to {text_mean_path}")
    print(f"Saved img_embed_mean to {x_mean_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'audio'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args.dataset_type)