import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from omegaconf import OmegaConf

from ..utils import get_2d_sincos_pos_embed

"""
Adapted from: https://github.com/facebookresearch/mae/blob/main/models_mae.py
"""

class ViTDecoder(nn.Module):
    def __init__(self, cfg, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(ViTDecoder, self).__init__()
        
        ## Adapted from MAE official repo
        self.patch_size = cfg.decoder.patch_size
        self.in_chans = cfg.decoder.in_chans
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder.embed_dim))
        
        self.num_patches = int(cfg.decoder.img_size ** 2 / self.patch_size ** 2)
        
        self.decoder_embed = nn.Linear(cfg.encoder.embed_dim, cfg.decoder.embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, 
                                                          int(cfg.decoder.embed_dim)), requires_grad=False) 
        
        self.decoder_blocks = nn.ModuleList([
            Block(cfg.decoder.embed_dim, cfg.decoder.num_heads, cfg.decoder.mlp_ratio, qkv_bias=True, 
                  norm_layer=norm_layer)
            for i in range(cfg.decoder.depth)])

        self.decoder_norm = norm_layer(cfg.decoder.embed_dim)
        self.decoder_pred = nn.Linear(cfg.decoder.embed_dim, self.patch_size**2 * self.in_chans, bias=True) # decoder to patch
        
        if not OmegaConf.is_none(cfg.decoder, "checkpoint"):
            self.load_from_checkpoint(cfg)
    
    def load_from_checkpoint(self, cfg):
        print(cfg.decoder.checkpoint)
        ckpt = torch.load(cfg.decoder.checkpoint)
        
        msg = self.load_state_dict(ckpt['model'], strict=False)

        print('='*80)
        print(msg)
        print('='*80)
        
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                    int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
        
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
        
        
    def forward(self, x):
        # x shape: [bs, prefix_length, embed_dim]
        
        if x.shape[1] > self.num_patches: # truncate
            x = x[:, self.num_patches, :]
        elif x.shape[1] < self.num_patches: # append mask tokens
            mask_tokens = self.mask_token.repeat(x.shape[0], self.num_patches - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) # (batch_size, num_patches, patch_size^2 * num_channels)
        
        # unpatchify
        x = self.unpatchify(x)
        
        return x