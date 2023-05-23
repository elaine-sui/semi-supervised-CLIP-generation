import torch
import pickle
import argparse

from tqdm import tqdm

from ..enums import DatasetType

TEXT_TO_VID_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_vid_gap.pkl'
TEXT_TO_MED_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_med_img_gap.pkl'
TEXT_TO_AMINO_ACID_GAP_PATH = '/pasteur/u/esui/data/c3/normalized_cap_to_amino_acid_gap.pkl'

def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        data_path =  '/pasteur/u/esui/data/c3/data_videoclip_train.pkl'
        gap_path = TEXT_TO_VID_GAP_PATH
    elif dataset_type == DatasetType.Medical:
        data_path =  '/pasteur/u/esui/data/c3/data_convirt_train.pkl'
        gap_path = TEXT_TO_MED_GAP_PATH
    elif dataset_type == DatasetType.Amino_Acid:
        data_path =  '/pasteur/u/esui/data/c3/data_clasp_train.pkl'
        gap_path = TEXT_TO_AMINO_ACID_GAP_PATH
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")
    
    return data_path, gap_path


def main(dataset_type):
    data_path, gap_path = get_data_paths(dataset_type)
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    embed_gap_sum = torch.zeros((1, 512)) # embed dim
        
    for item in tqdm(all_data):
        cap_embed = item['y_embed']
        x_embed = item['x_embed']
        embed_gap = (x_embed / x_embed.norm()) - (cap_embed / cap_embed.norm())
        
        embed_gap_sum += embed_gap
    
    embed_gap_avg = embed_gap_sum / len(all_data)
    
    with open(gap_path, 'wb') as f:
        pickle.dump(embed_gap_avg, f)
    
    print(f"Saved cap_to_x_gap to {gap_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['video', 'medical', 'amino_acid'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args.dataset_type)