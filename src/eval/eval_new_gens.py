import json
import os

from src.eval.utils.caption_evaluate import evaluate_on_coco_caption

NUM_REPEATS = 5

def reformat_preds(raw_pred_file, dir):
    """
    Reformat predictions from list(zip([generations, originals])) to list({"image_id": ..., "caption": ..., "id": ...})
    """

    filename = os.path.split(raw_pred_file)[1][:-5]

    with open(raw_pred_file, "r") as f:
        gens, refs = json.load(f)
    gens, refs = gens[0], refs[0]
    gens = [gen.replace("<|endoftext|>", "").strip() for gen in gens]

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
    for i in range(len(gens) // NUM_REPEATS):
        gen = gens[i*NUM_REPEATS]

        entry = {"image_id": i, "caption": gen, "id": i}
        preds.append(entry)
    
    new_pred_file = f'{dir}/{filename}_converted_preds.json'

    with open(new_pred_file, 'w') as f:
        json.dump(preds, f, indent=4)

    print(f"New pred file: {new_pred_file}")

    return new_pred_file, new_labels_file


def run_eval(raw_pred_file, out_file=None, dir=None):
    new_pred_file, new_labels_file = reformat_preds(raw_pred_file, dir=dir)

    metrics_dict = evaluate_on_coco_caption(new_pred_file, new_labels_file, out_file)

    print(metrics_dict)
    print(f"Metrics file at {out_file}")