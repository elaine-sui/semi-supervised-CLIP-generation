import torch
import pickle
import os

from tqdm import tqdm

TEXT_TO_IMG_GAP_PATH = '/pasteur/u/esui/data/coco/normalized_cap_to_img_gap.pkl'

def main():
    data_path =  '/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_train+restval.pkl'
    
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # {image_id: {"img_path": ..., "embed": ...}}
    images = all_data["images"]
    # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
    captions = all_data["captions"]
    
    if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
        print("=> Loading caption_id_2_image_id, captions_tokens, all_len dicts")
        with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
            captions_tokens, caption_id_2_image_id, all_len = pickle.load(f)
    
    num_captions = len(captions)
    embed_gap_sum = torch.zeros((1, 512)) # clip embed dim
        
    for cap_id in tqdm(captions):
        img_id = caption_id_2_image_id[cap_id]
        cap_embed = captions[cap_id]["embed"]
        img_embed = images[img_id]["embed"]
        embed_gap = (img_embed / img_embed.norm()) - (cap_embed / cap_embed.norm())
        
        embed_gap_sum += embed_gap
    
    embed_gap_avg = embed_gap_sum / num_captions
    
    with open(TEXT_TO_IMG_GAP_PATH, 'wb') as f:
        pickle.dump(embed_gap_avg, f)
    
    print(f"Saved cap_to_img_gap to {TEXT_TO_IMG_GAP_PATH}")
    

if __name__ == '__main__':
    main()