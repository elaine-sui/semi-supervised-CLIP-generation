import pickle
import argparse
import os
import random
import sys

sys.path.append(os.getcwd())
from src.enums import DatasetType

def get_data_paths(dataset_type):
    output_folder = '/pasteur/u/esui/data/c3'

    if dataset_type == DatasetType.Video:
        data_path = '/pasteur/u/yuhuiz/compositional_generation/ImageBind/data_video_msrvtt_imagebind.pkl'
        # data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_videoclip.pkl'
        # data_path = '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_videoclip_3k.pkl'
    elif dataset_type == DatasetType.Medical:
        # data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_convirt.pkl'
        # data_path = '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_convirt_10k.pkl'
        # data_path = '/pasteur/u/esui/data/c3/data_medclip_10k.pkl'
        data_path = '/pasteur/u/esui/data/c3/data_medclip_no_aug_10k.pkl'
    elif dataset_type == DatasetType.Amino_Acid:
        data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_clasp.pkl'
    elif dataset_type == DatasetType.Audio:
        data_path = '/pasteur/u/yuhuiz/compositional_generation/ImageBind/data_audio_clotho_imagebind.pkl'
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")

    return data_path, output_folder


def main(args):
    data_path, output_folder = get_data_paths(args.dataset_type)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    ids = list(range(len(all_data)))

    if 'split' in all_data[0]:
        train_data = [d for d in all_data if d['split'] == 'train']
        val_data = [d for d in all_data if d['split'] == 'test']
        
        if len(val_data) > 5000:
            train_data.extend(val_data[:-5000])
            val_data = val_data[-5000:]
        
        train_data = dict(zip(ids[:len(train_data)], train_data))
        val_data = dict(zip(ids[len(train_data):], val_data))
        test_data = val_data
    else:
        # Shuffle -- seeded
        random.Random(args.seed).shuffle(all_data)

        # Split
        num_train = int(len(all_data) * args.train_ratio)
        train_data = all_data[:num_train]
        train_data = dict(zip(ids[:num_train], train_data))

        num_test = int(len(all_data) * args.test_ratio)
        test_data = all_data[-num_test:]
        test_data = dict(zip(ids[-num_test:], test_data))

        val_data = all_data[num_train:-num_test]
        val_data = dict(zip(ids[num_train:-num_test], val_data))

    # Save to pickle
    file_name = os.path.split(data_path)[1]

    for split, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        new_file_path = os.path.join(output_folder, file_name[:-4] + f"_{split}" + ".pkl")

        with open(new_file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Dumped {split} data at: {new_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'medical', 'amino_acid', 'audio'])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")
    print(f"Train ratio: {args.train_ratio}")

    main(args)
