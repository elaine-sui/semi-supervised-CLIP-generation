import pickle
import argparse
import os
import sys

sys.path.append(os.getcwd())
from src.enums import DatasetType

def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        val_data_path = '/pasteur/u/esui/data/c3/data_videoclip_3k_val.pkl'
        test_data_path = '/pasteur/u/esui/data/c3/data_videoclip_3k_test.pkl'
        
    elif dataset_type == DatasetType.Medical:
        val_data_path = '/pasteur/u/esui/data/c3/data_medclip_no_aug_10k_val.pkl'
        test_data_path = '/pasteur/u/esui/data/c3/data_medclip_no_aug_10k_test.pkl'
    elif dataset_type == DatasetType.Amino_Acid:
        val_data_path = '/pasteur/u/esui/data/c3/data_clasp_val.pkl'
        test_data_path = '/pasteur/u/esui/data/c3/data_clasp_test.pkl'
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")

    return val_data_path, test_data_path


def main(args):
    val_path, test_path = get_data_paths(args.dataset_type)

    print(f"Loading data from {val_path}")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    print(f"Loading data from {test_path}")
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    val_data.update(test_data)

    val_test_path = val_path[:-4] + '_test.pkl'
    print(f"Dumping combined val+test to {val_test_path}")

    with open(val_test_path, 'wb') as f:
        pickle.dump(val_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'medical', 'amino_acid'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")

    main(args)
