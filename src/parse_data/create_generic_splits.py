import pickle
import argparse
import os
import random

from ..enums import DatasetType

def get_data_paths(dataset_type):
    output_folder = '/pasteur/u/esui/data/c3'

    if dataset_type == DatasetType.Video:
        data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_videoclip.pkl'
    elif dataset_type == DatasetType.Medical:
        data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_convirt.pkl'
    elif dataset_type == DatasetType.Amino_Acid:
        data_path =  '/pasteur/u/yuhuiz/archive/neurips_modality_gap/pull_figure/data_clasp.pkl  '
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")

    return data_path, output_folder


def main(args):
    data_path, output_folder = get_data_paths(args.dataset_type)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    # Shuffle -- seeded
    random.Random(args.seed).shuffle(all_data)

    # Split
    num_train = int(len(all_data) * args.train_ratio)
    train_data = all_data[:num_train]

    num_test = int(len(all_data) * args.test_ratio)
    test_data = all_data[-num_test:]

    val_data = all_data[num_train:-num_test]

    # Save to pickle
    file_name = os.path.split(data_path)[0]

    for split, data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        new_file_path = os.path.join(output_folder, file_name[:-4] + f"_{split}" + ".pkl")

        with open(new_file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Dumped {split} data at: {new_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['video', 'medical', 'amino_acid'])
    parser.add_argument('--train_ratio', type=0.8)
    parser.add_argument('--test_ratio', type=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")
    print(f"Train ratio: {args.train_ratio}")

    main(args)
