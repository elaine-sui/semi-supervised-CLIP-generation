import json
import os

from src.eval.utils.caption_evaluate import evaluate_on_coco_caption

def reformat_preds(raw_pred_file, refs_json, dir):
    """
    Reformat predictions from list(zip([generations, originals])) to list({"image_id": ..., "caption": ..., "id": ...})
    """
    filename = os.path.split(raw_pred_file)[1][:-5]

    with open(raw_pred_file, "r") as f:
        gens, refs = json.load(f)
    gens, refs = gens[0], refs[0]
    gens = [gen.replace("<|endoftext|>", "").strip() for gen in gens]

    # Convert anns into dict of ref2img_id
    ref2img_id_path = f"{dir}/ref2img_id.json"

    if os.path.exists(ref2img_id_path):
        with open(ref2img_id_path, 'r') as f:
            ref2img_id = json.load(f)
    else:
        with open(refs_json, "r") as f:
            anns = json.load(f)["annotations"]

        ref2img_id = {}
        for ann in anns:
            ref = ann['caption']
            img_id = ann['image_id']
            ref2img_id[ref] = img_id

        with open(ref2img_id_path, 'w') as f:
            json.dump(ref2img_id, f, indent=4)
    

    preds = []
    img_ids = []
    for i, gen in enumerate(gens):
        # find ref caption entry, get corresponding prediction
        ref = refs[i]
        img_id = ref2img_id[ref]

        if img_id in img_ids: # because an error for some reason, multiple captions for a single image used as GT
            continue
        else:
            img_ids.append(img_id)

        entry = {"image_id": img_id, "caption": gen, "id": i}

        preds.append(entry)
    
    new_pred_file = f'{dir}/{filename}_converted_preds.json'

    with open(new_pred_file, 'w') as f:
        json.dump(preds, f, indent=4)

    print(f"New pred file: {new_pred_file}")

    return new_pred_file


def run_eval(raw_pred_file, labels_json, out_file=None, dir=None):
    new_pred_file = reformat_preds(raw_pred_file, labels_json, dir=dir)

    metrics_dict = evaluate_on_coco_caption(new_pred_file, labels_json, out_file)

    print(metrics_dict)
    print(f"Metrics file at {out_file}")