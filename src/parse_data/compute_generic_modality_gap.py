import torch
import pickle
import argparse

from tqdm import tqdm

import os
import sys
sys.path.append(os.getcwd())
from src.enums import DatasetType
# from ..enums import DatasetType

TEXT_TO_VID_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_vid_gap.pkl'
TEXT_TO_MED_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_med_img_gap.pkl'
TEXT_TO_AMINO_ACID_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_amino_acid_gap.pkl'
TEXT_TO_AUDIO_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_audio_gap.pkl'

def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        data_path =  '/pasteur/u/esui/data/c3/data_videoclip_train.pkl'
        gap_path = TEXT_TO_VID_GAP_PATH
    elif dataset_type == DatasetType.Medical:
        # data_path =  '/pasteur/u/esui/data/c3/data_convirt_train.pkl'
        data_path =  '/pasteur/u/esui/data/c3/data_medclip_no_aug_10k_train.pkl'
        gap_path = TEXT_TO_MED_GAP_PATH
    elif dataset_type == DatasetType.Amino_Acid:
        data_path =  '/pasteur/u/esui/data/c3/data_clasp_train.pkl'
        gap_path = TEXT_TO_AMINO_ACID_GAP_PATH
    elif dataset_type == DatasetType.Audio:
        data_path = '/pasteur/u/esui/data/c3/data_audio_clotho_imagebind_train.pkl'
        gap_path = TEXT_TO_AUDIO_GAP_PATH
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")
    
    return data_path, gap_path


def main(dataset_type):
    data_path, gap_path = get_data_paths(dataset_type)
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    embed_dim = all_data[0]['x_embed'].shape[0]
    embed_gap_sum = torch.zeros((1, embed_dim)) # embed dim
        
    for id, item in tqdm(all_data.items()):
        cap_embed = torch.from_numpy(item['y_embed'])
        x_embed = torch.from_numpy(item['x_embed'])
        embed_gap = (x_embed / x_embed.norm()) - (cap_embed / cap_embed.norm())
        
        embed_gap_sum += embed_gap
    
    embed_gap_avg = embed_gap_sum / len(all_data)
    
    with open(gap_path, 'wb') as f:
        pickle.dump(embed_gap_avg, f)
    
    print(f"Saved cap_to_x_gap to {gap_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'medical', 'amino_acid', 'audio'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args.dataset_type)