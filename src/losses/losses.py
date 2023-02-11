import torch
import torch.nn as nn

def ce_caption_loss(logits, labels):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                       labels.flatten(), ignore_index=0)

def image_reconstruction_loss(pred, original_images, model):
    # patchify 
    target = model.model.patchify(original_images)
    
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    return loss.mean()