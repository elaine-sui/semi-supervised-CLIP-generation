import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Optional

from .vit_decoder import ViTDecoder
from .builder import build_huggingface_model
from .mlp import MLP
from ..enums import Modality
from ..utils import get_2d_sincos_pos_embed ## need to check this for vision!
from ..eval import generate2

"""
Code adapted from: https://github.com/rmokady/CLIP_prefix_caption/blob/main/train.py
"""
        
class Decoder(nn.Module):
    
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        self.modality = cfg.decoder.modality.lower()
        
        if cfg.decoder.modality.lower() == Modality.Vision:
            self.model = ViTDecoder(cfg)
            self.embed_size = cfg.decoder.embed_dim
        elif cfg.decoder.modality.lower() == Modality.Language:
            self.model = build_huggingface_model(cfg.decoder.model.lower())
            self.embed_size = self.model.transformer.wte.weight.shape[1]
        else:
            raise NotImplementedError(
                f"Decoder modality not implemented for {cfg.decoder.modality.lower()}"
            )
        
        self.prefix_length = cfg.model.prefix_length
        
        prefix_size = 640 if cfg.model.is_rn else 512
        
        self.clip_project = MLP((prefix_size, (self.embed_size * self.prefix_length) // 2,
                                     self.embed_size * self.prefix_length))
    
        if not OmegaConf.is_none(cfg.model, "checkpoint"):
            self.load_from_checkpoint(cfg)
    
    def load_from_checkpoint(self, cfg):
        print(f"=> Loading decoder checkpoint from {cfg.model.checkpoint}")
        ckpt = torch.load(cfg.model.checkpoint)
        
        ckpt = {k.replace('gpt', 'model'):v for k,v in ckpt.items()}
        msg = self.load_state_dict(ckpt, strict=False)

        print('='*80)
        print(msg)
        print('='*80)
    
    
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    def forward_language(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        
        embedding_text = self.model.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.embed_size) 
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        out = self.model(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out    
    
    def forward_vision(self, x):
        x = self.clip_project(x).view(-1, self.prefix_length, self.embed_size)
        x = self.model(x)

        return x
    
    def forward(self, **kwargs):
        if self.modality == Modality.Language:
            return self.forward_language(**kwargs)
        elif self.modality == Modality.Vision:
            return self.forward_vision(**kwargs)
        
        