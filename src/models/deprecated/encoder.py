import torch
import torch.nn as nn
from ..enums import Modality
from ..builder import build_huggingface_model
from mlp import MLP


class Encoder(nn.Module):
    
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        
        self.modality = cfg.encoder.modality
        self.clip_model_name = cfg.encoder.clip_model_type.replace('/', '_')
        self.clip_model, _ = clip.load(cfg.encoder.clip_model_type, device=device, jit=False)
    
    def forward(self, x):
        with torch.no_grad(): ## TODO: Maybe move this out when parsing dataset and saving embeddings?
            if self.modality == Modality.Vision:
                return self.clip_model.encode_image(x).cpu()
            elif self.modality == Modality.Language:
                return self.clip_model.encode_text(x).cpu()
        
        