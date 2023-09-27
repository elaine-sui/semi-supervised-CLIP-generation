import collections
import json
import os
import yaml
import pandas as pd
import evaluate

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')

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

def get_pred_filename(output_dir, split, epoch=None, rank=0, temp=False):
    if epoch:
        if temp:
            return os.path.join(output_dir, f'{split}_epoch_{epoch}_pred_rank{rank}_temp.pkl')
        return os.path.join(output_dir, f'{split}_epoch_{epoch}_pred_rank{rank}.json')
    else:
        if temp:
            return os.path.join(output_dir, f'{split}_pred_rank{rank}_temp.pkl')
        return os.path.join(output_dir, f'{split}_pred_rank{rank}.json')

def get_metrics_out_filename(output_dir, split, epoch=None):
    if epoch:
        return os.path.join(output_dir, f'{split}_epoch_{epoch}_metrics.json')
    else:
        return os.path.join(output_dir, f'{split}_metrics.json')


def add_predictions_to_results_json(predictions, filepath):
    
    parent_dir = os.path.pardir(filepath)
    os.makedirs(parent_dir, exist_ok=True)
    # filepath = get_pred_filename(output_dir, split, epoch, rank)
    all_preds = []
    # if os.path.exists(filepath):
    #     with open(filepath, 'r') as f:
    #         all_preds = json.load(f)
        
    all_preds.extend(predictions)

    with open(filepath, 'w') as f:
        json.dump(predictions, f, indent=4)
        
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

def evaluate_list(gens, refs):
    rouge_score = rouge.compute(predictions=gens, references=refs)
    bleu_score = bleu.compute(predictions=gens, references=refs)
    meteor_score = meteor.compute(predictions=gens, references=refs)
    metrics_dict = {}
    metrics_dict.update(rouge_score)
    metrics_dict['bleu'] = bleu_score['bleu']
    metrics_dict['meteor'] = meteor_score['meteor']

    return metrics_dict
    
        