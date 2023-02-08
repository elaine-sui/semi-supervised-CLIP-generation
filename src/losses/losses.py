import torch
import torch.nn as nn

def ce_caption_loss(logits, labels):
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                                       labels.flatten(), ignore_index=0)

def image_reconstruction_loss(pred, original_image):
    return nn.functional.mse_loss(pred, original_image)