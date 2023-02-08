import os
import json
from utils.caption_evaluate import evaluate_on_coco_caption

def test_eval_perfect():
    predictions = [
        {"image_id": 2, "caption": "A child holding a flowered umbrella and petting a yak.", "id": 474921},
        {"image_id": 60, "caption": "A lady riding her bicycle on the side of a street.", "id": 578262},
        # {"image_id": 82, "caption": "A toilet sitting in an outdoor area with a helmet resting on top of it.", "id": 349429}
    ]
    
    labels = [
        {"image_id": 2, "caption": "A child holding a flowered umbrella and petting a yak.", "id": 474921},
        {"image_id": 60, "caption": "A lady riding her bicycle on the side of a street.", "id": 578262},
        {"image_id": 82, "caption": "A toilet sitting in an outdoor area with a helmet resting on top of it.", "id": 349429},
        {"image_id": 10, "caption": "A toilet.", "id": 349477}
    ]
    
    os.makedirs('test_files', exist_ok=True)
    
    res_filename = 'test_files/test_eval_perfect.json'
    label_filename = 'test_files/test_labels.json'
    out_filename = 'test_files/test_out.json'
    
    with open(res_filename, 'w') as f:
        json.dump(predictions, f)
        
    with open(label_filename, 'w') as f:
        json.dump({"annotations" : labels, "images": [{"id": 2}, {"id": 60} ,{"id": 82}]}, f)
        
    metrics = evaluate_on_coco_caption(res_filename, label_filename, out_filename)
    print(metrics)
    

def test_eval_error():
    predictions = [
        {"image_id": 2, "caption": "A child holding a flowered umbrella and petting a yak.", "id": 474921},
        {"image_id": 60, "caption": "A lady riding her bicycle on the side of a street.", "id": 578262},
        {"image_id": 82, "caption": "A chair in a garden.", "id": 349429}
    ]
    
    labels = [
        {"image_id": 2, "caption": "A child holding a flowered umbrella and petting a yak.", "id": 474921},
        {"image_id": 60, "caption": "A lady riding her bicycle on the side of a street.", "id": 578262},
        {"image_id": 82, "caption": "A toilet sitting in an outdoor area with a helmet resting on top of it.", "id": 349429},
    ]
    
    os.makedirs('test_files', exist_ok=True)
    res_filename = 'test_files/test_eval_error.json'
    label_filename = 'test_files/test_labels.json'
    out_filename = 'test_files/test_out.json'
    
    with open(res_filename, 'w') as f:
        json.dump(predictions, f)
        
    with open(label_filename, 'w') as f:
        json.dump({"annotations" : labels, "images": [{"id": 2}, {"id": 60} ,{"id": 82}]}, f)
        
    metrics = evaluate_on_coco_caption(res_filename, label_filename, out_filename)
    print(metrics)

if __name__ == '__main__':
    test_eval_perfect()
    
    test_eval_error()