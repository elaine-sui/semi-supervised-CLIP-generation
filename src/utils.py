import collections
import json
import os
import yaml
import pandas as pd

def flatten(d, parent_key="", sep="."):
    """flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_pred_filename(output_dir, split):
    return os.path.join(output_dir, f'{split}_pred.json')

def get_metrics_out_filename(output_dir, split):
    return os.path.join(output_dir, f'{split}_metrics.json')


def add_predictions_to_results_json(predictions, output_dir, split):
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = get_pred_filename(output_dir, split)
    all_preds = []
    # if os.path.exists(filepath):
    #     with open(filepath, 'r') as f:
    #         all_preds = json.load(f)
        
    all_preds.extend(predictions)

    with open(filepath, 'w') as f:
        json.dump(predictions, f)
        
    print(f"=> Predictions at {filepath}")

def get_best_ckpt_path(ckpt_paths, ascending=False):
    """get best ckpt path from a list of ckpt paths

    ckpt_paths: JSON file with ckpt path to metric pair
    ascending: sort paths based on ascending or descending metrics
    """

    with open(ckpt_paths, "r") as stream:
        ckpts = yaml.safe_load(stream)

    ckpts_df = pd.DataFrame.from_dict(ckpts, orient="index").reset_index()
    ckpts_df.columns = ["path", "metric"]
    best_ckpt_path = (
        ckpts_df.sort_values("metric", ascending=ascending).head(1)["path"].item()
    )

    return best_ckpt_path
    
        
