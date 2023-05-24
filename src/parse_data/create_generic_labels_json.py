import json
from tqdm import tqdm
import pickle
import argparse
import sys
import os

sys.path.append(os.getcwd())
from src.enums import DatasetType

DATA_ROOT = '/pasteur/u/esui'
splits = ["train", "val", "test"]

def get_label_json_list(dataset_type):
    labels_jsons_lst = dict(zip(splits, 
                            [f"{DATA_ROOT}/data/c3/annotations/{dataset_type}_labels_{split}.json" for split in splits]))

    return labels_jsons_lst


def get_data_paths(dataset_type):
    if dataset_type == DatasetType.Video:
        train_data_path =  '/pasteur/u/esui/data/c3/data_videoclip_train.pkl'
        val_data_path =  '/pasteur/u/esui/data/c3/data_videoclip_val.pkl'
        test_data_path =  '/pasteur/u/esui/data/c3/data_videoclip_test.pkl'
    elif dataset_type == DatasetType.Medical:
        train_data_path =  '/pasteur/u/esui/data/c3/data_convirt_train.pkl'
        val_data_path =  '/pasteur/u/esui/data/c3/data_convirt_val.pkl'
        test_data_path =  '/pasteur/u/esui/data/c3/data_convirt_test.pkl'
    elif dataset_type == DatasetType.Amino_Acid:
        train_data_path =  '/pasteur/u/esui/data/c3/data_clasp_train.pkl'
        val_data_path =  '/pasteur/u/esui/data/c3/data_clasp_val.pkl'
        test_data_path =  '/pasteur/u/esui/data/c3/data_clasp_test.pkl'
    else:
        raise NotImplementedError(f"dataset type {dataset_type} not implemented")
    
    return train_data_path, val_data_path, test_data_path

def create_labels_json(dataset_type):
    all_labels = dict(zip(splits, [
        {"annotations": [], "images": []},
        {"annotations": [], "images": []},
        {"annotations": [], "images": []},
        {"annotations": [], "images": []}]))
    
    labels_jsons_lst = get_label_json_list(dataset_type)
    
    out_paths = labels_jsons_lst

    labels_dir = os.path.split(labels_jsons_lst['train'])[0]
    os.makedirs(labels_dir, exist_ok=True)

    train_data_path, val_data_path, test_data_path = get_data_paths(dataset_type)
    
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)

    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    for split, data in zip(splits, [train_data, val_data, test_data]):
        for i, id in tqdm(enumerate(data)):
            caption = data[id]['y']
            
            image_dict = {"id": id}
            all_labels[split]["images"].append(image_dict)

            caption_dict = {"image_id": id, "caption": caption, "id": id}
            all_labels[split]["annotations"].append(caption_dict)
            
    for split in splits:
        out_path = out_paths[split]
        with open(out_path, 'w') as f:
            json.dump(all_labels[split], f)

    print("Total number of annotations")
    print_anns_totals(all_labels)
        
def print_anns_totals(all_labels):
    anns_totals = [len(all_labels[split]['annotations']) for split in splits]
    print(dict(zip(splits, anns_totals)))
    print("Total number of images")
    imgs_totals = [len(all_labels[split]['images']) for split in splits]
    print(dict(zip(splits, imgs_totals)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='video', choices=['video', 'medical', 'amino_acid'])
    args = parser.parse_args()

    print(f"Dataset type: {args.dataset_type}")
    create_labels_json(args.dataset_type)