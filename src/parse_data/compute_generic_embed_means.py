import torch
import pickle
import argparse

from tqdm import tqdm

import os
import sys
sys.path.append(os.getcwd())
from src.enums import DatasetType
# from ..enums import DatasetType

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

def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        data_path =  '/pasteur/u/esui/data/c3/data_videoclip_3k_train.pkl'
        text_mean_path = TEXT_VIDEOCLIP_EMBED_MEAN
        x_mean_path = VIDEO_EMBED_MEAN
    elif dataset_type == DatasetType.Medical:
        # data_path =  '/pasteur/u/esui/data/c3/data_convirt_10k_train.pkl'
        # text_mean_path = TEXT_CONVIRT_EMBED_MEAN
        # x_mean_path = MED_IMAGES_EMBED_MEAN
        # data_path = '/pasteur/u/esui/data/c3/data_medclip_10k_train.pkl'
        data_path = '/pasteur/u/esui/data/c3/data_medclip_no_aug_10k_train.pkl'
        text_mean_path = TEXT_MEDCLIP_NO_AUG_EMBED_MEAN
        x_mean_path = MED_IMAGES_MEDCLIP_NO_AUG_EMBED_MEAN
    elif dataset_type == DatasetType.Amino_Acid:
        data_path =  '/pasteur/u/esui/data/c3/data_clasp_train.pkl'
        text_mean_path = TEXT_CLASP_EMBED_MEAN
        x_mean_path = AMINO_ACID_EMBED_MEAN
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")
    
    return data_path, text_mean_path, x_mean_path


def main(dataset_type):
    data_path, text_mean_path, x_mean_path = get_data_paths(dataset_type)
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    embed_dim = all_data[0]['x_embed'].shape[0]
    text_mean = torch.zeros((1, embed_dim)) # embed dim
    x_mean = torch.zeros((1, embed_dim)) # embed dim
        
    for id, item in tqdm(all_data.items()):
        text_embed = torch.from_numpy(item['y_embed'])
        x_embed = torch.from_numpy(item['x_embed'])
        text_mean += text_embed / text_embed.norm()
        x_mean += x_embed / x_embed.norm()
    
    text_mean /= len(all_data)
    x_mean /= len(all_data)
    
    with open(text_mean_path, 'wb') as f:
        pickle.dump(text_mean, f)
    
    with open(x_mean_path, 'wb') as f:
        pickle.dump(x_mean, f)
    
    print(f"Saved text_embed_mean to {text_mean_path}")
    print(f"Saved x_embed_mean to {x_mean_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'medical', 'amino_acid'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args.dataset_type)