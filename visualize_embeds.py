import pickle
import random
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.manifold import TSNE

from src.parse_data import TEXT_EMBED_MEAN, IMAGE_EMBED_MEAN
from src.enums import Modality

train_restval_data_path = '/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_train+restval.pkl'

def sample_embeds(modality, n):
    if modality == Modality.Vision:
        embed_mean_path = IMAGE_EMBED_MEAN
        input_type = "images"
    else:
        embed_mean_path = TEXT_EMBED_MEAN
        input_type = "captions"
    
    print("=> Loading mean")
    with open(embed_mean_path, 'rb') as f:
        embed_mean = pickle.load(f)
    
    data = all_data[input_type]
    
    indices = random.sample(list(data.keys()), n)
    
    embeds = [data[idx]["embed"].squeeze().numpy() for idx in indices]
    embeds = np.stack(embeds)
    embeds = embeds / np.linalg.norm(embeds, axis=1).reshape(-1, 1)
    
    return embeds, embed_mean


def plot(n=100):
    embeds_language, embed_mean_language = sample_embeds(Modality.Language, n)
    embeds_vision, embed_mean_vision = sample_embeds(Modality.Vision, n)
    
    tsne = TSNE()
    
    embeds = np.vstack([embeds_language, embeds_vision])
    embeds_removed_mean = np.vstack([embeds_language - embed_mean_language.numpy(),
                                    embeds_vision - embed_mean_vision.numpy()])
    
    two_dimensional_embeds = tsne.fit_transform(embeds)
    
    plt.clf()
    plt.figure()
    plt.title(f"normed embeds")
    plt.scatter(two_dimensional_embeds[:100, 0], two_dimensional_embeds[:100, 1], color='red')
    plt.scatter(two_dimensional_embeds[100:, 0], two_dimensional_embeds[100:, 1], color='blue')
    plt.savefig(f"output/normed_embeds.png")
    
    two_dimensional_embeds = tsne.fit_transform(embeds_removed_mean)
    
    plt.clf()
    plt.figure()
    plt.title(f"normed embeds removed mean")
    plt.scatter(two_dimensional_embeds[:100, 0], two_dimensional_embeds[:100, 1], color='red')
    plt.scatter(two_dimensional_embeds[100:, 0], two_dimensional_embeds[100:, 1], color='blue')
    plt.savefig(f"output/normed_embeds_remove_mean.png")
    
def compute_residual(n=200):
    embeds_language, embed_mean_language = sample_embeds(Modality.Language, n)
    embeds_vision, embed_mean_vision = sample_embeds(Modality.Vision, n)
    
    gap = (embeds_language - embed_mean_language.numpy()) - (embeds_vision - embed_mean_vision.numpy())
    
    print(f"Norm of gap after subtracting means: {np.linalg.norm(gap)}")
    

if __name__ == '__main__':
    print("=> Loading data")
    with open(train_restval_data_path, 'rb') as f:
        all_data = pickle.load(f)
        
    # plot()
    compute_residual()
    
    
    
    