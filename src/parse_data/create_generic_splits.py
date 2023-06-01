import pickle
import argparse
import os
import random
import sys
from tqdm import tqdm
import torch

sys.path.append(os.getcwd())
from src.enums import DatasetType

def get_data_paths(dataset_type):
    output_folder = '/pasteur/u/esui/data/c3'

    if dataset_type == DatasetType.Video:
        data_path = '/pasteur/u/yuhuiz/iccv_c3/data_video_msrvtt_imagebind.pkl'
    elif dataset_type == DatasetType.Audio:
        data_path = '/pasteur/u/yuhuiz/iccv_c3/data_audio_clotho_imagebind.pkl'
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")

    return data_path, output_folder

# captions -- {caption_id: {caption_raw: .., input_id: ..}}
# embeddings -- {input_id: embedding}

splits2 = ['train', 'val', 'test']

def main(args):
    data_path, output_folder = get_data_paths(args.dataset_type)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f) # list of [x, x_embed, y, y_embed, split] (x's may be repeated)
    
    unique_inputs = list(set([d['x'] for d in data]))
    unique_inputs_to_ids = dict(zip(unique_inputs, range(len(unique_inputs))))

    all_inputs = dict(zip(splits2, [{}, {}, {}]))
    all_captions = dict(zip(splits2, [{}, {}, {}]))
    for caption_id in tqdm(range(len(data))):
        d = data[caption_id]
        split, x, x_embed, y, y_embed = d["split"], d["x"], d["x_embed"], d["y"], d["y_embed"]

        # Get and save image and image embed
        input_id = unique_inputs_to_ids[x]
        
        ## Note: changes for all the keys!
        all_inputs[split][input_id] = {"file_name": x, "embed": torch.from_numpy(x_embed)}
        
        all_captions[split][caption_id] = {
            "caption": y,
            "input_id": input_id,
            "embed": torch.from_numpy(y_embed)
        }
    
    if len(all_inputs['val']) == 0:
        all_inputs["val"] = all_inputs["test"]
        all_captions["val"] = all_captions["test"]

    file_name = os.path.split(data_path)[1]
    for split in splits2:
        new_file_path = os.path.join(output_folder, file_name[:-4] + f"_{split}" + ".pkl")
        with open(new_file_path, 'wb') as f:
            pickle.dump({"inputs": all_inputs[split],
                         "captions": all_captions[split]}, f)

    print('Done')
    print_totals(all_inputs, all_captions)
    return 0


def print_totals(all_inputs, all_captions):
    print('Done')
    print("Total number of inputs (so far)")
    embed_totals = [len(all_inputs[split].values()) for split in splits2]
    print(dict(zip(splits2, embed_totals)))
    print("Total number of captions (so far)")
    caption_totals = [len(all_captions[split].values()) for split in splits2]
    print(dict(zip(splits2, caption_totals)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'audio'])
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args)
