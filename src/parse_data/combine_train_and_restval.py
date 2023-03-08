import pickle

if __name__ == '__main__':
    train_data_path = './data/coco/oscar_split_ViT-B_32_train.pkl'
    restval_data_path = './data/coco/oscar_split_ViT-B_32_restval.pkl'
    train_restval_data_path = './data/coco/oscar_split_ViT-B_32_train+restval.pkl'
    
    print("Loading train data")
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    
    print("Loading restval data")
    with open(restval_data_path, 'rb') as f:
        restval_data = pickle.load(f)
        
    train_images = train_data['images']
    restval_images = restval_data['images']
    
    train_captions = train_data['captions']
    restval_captions = restval_data['captions']
    
    train_images.update(restval_images)
    train_captions.update(restval_captions)
    
    print("Saving train+restval data")
    with open(train_restval_data_path, 'wb') as f:
        pickle.dump({"images": train_images, "captions": train_captions}, f)
    
