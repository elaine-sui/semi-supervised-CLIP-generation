import json
import os
import re
import pickle

from src.eval.utils.caption_evaluate import evaluate_on_coco_caption

NUM_REPEATS = 5

def reformat_preds(raw_pred_file, dir, full_refs_path):
    """
    Reformat predictions from list(zip([generations, originals])) to list({"image_id": ..., "caption": ..., "id": ...})
    """

    filename = os.path.split(raw_pred_file)[1][:-5]

    # Load generations
    with open(raw_pred_file, "r") as f:
        gens, _ = json.load(f)
    gens = gens[0]
    
    # Clean generations
    sos = "<s>"
    eos = "</s>"

    processed_gens = []
    for gen in gens:
        if gen[:len(sos)] != sos:
            gen = ""
        elif eos in gen:
            gen = re.findall(r"<s>(.*?)</s>", gen)
            if len(gen) == 0: # nothing in between the tags
                gen = ""
            else:
                gen = gen[0].strip()
        else:
            gen = gen.replace(sos, "").strip()

        processed_gens.append(gen)

    gens = processed_gens

    # Load refs
    with open(full_refs_path, 'rb') as f:
        full_refs = pickle.load(f)
        refs = [item['y'] for item in full_refs if item["split"] == "test"]

    new_labels_file = os.path.join(dir, 'new_labels.json')

    if os.path.exists(new_labels_file):
        with open(new_labels_file, 'r') as f:
            labels = json.load(f)
    else:
        # Create new labels json file
        labels, images = [], []
        for i, ref in enumerate(refs):
            img_id = i // NUM_REPEATS
            entry = {"image_id": img_id, "caption": ref, "id": i}
            img_entry = {"id": img_id}
            labels.append(entry)
            images.append(img_entry)

        labels = {'annotations': labels, 'images': images}
        with open(new_labels_file, 'w') as f:
            json.dump(labels, f)

    preds = []
    for i, gen in enumerate(gens):
        entry = {"image_id": i, "caption": gen, "id": i}
        preds.append(entry)
    
    new_pred_file = f'{dir}/{filename}_converted_preds.json'

    with open(new_pred_file, 'w') as f:
        json.dump(preds, f, indent=4)

    print(f"=> New pred file: {new_pred_file}")

    return new_pred_file, new_labels_file


def run_eval(raw_pred_file, out_file=None, dir=None, full_refs_path=None):
    new_pred_file, new_labels_file = reformat_preds(raw_pred_file, dir=dir, full_refs_path=full_refs_path)

    metrics_dict = evaluate_on_coco_caption(new_pred_file, new_labels_file, out_file)

    print(metrics_dict)
    print(f"=> Metrics file at {out_file}")