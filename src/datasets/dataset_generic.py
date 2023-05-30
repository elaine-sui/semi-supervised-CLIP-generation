import torch
import os
import pickle
import sys
import random
import math

from typing import Tuple

import pytorch_lightning as pl
from transformers import GPT2Tokenizer
from omegaconf import OmegaConf

from ..enums import Modality, DatasetType
from ..parse_data import (
    TEXT_TO_VID_GAP_PATH, 
    TEXT_VIDEOCLIP_EMBED_MEAN, 
    VIDEO_EMBED_MEAN,
    TEXT_TO_MED_GAP_PATH,
    TEXT_CONVIRT_EMBED_MEAN,
    TEXT_MEDCLIP_EMBED_MEAN,
    MED_IMAGES_EMBED_MEAN,
    MED_IMAGES_MEDCLIP_EMBED_MEAN,
    TEXT_TO_AMINO_ACID_GAP_PATH,
    TEXT_CLASP_EMBED_MEAN,
    AMINO_ACID_EMBED_MEAN,
    TEXT_MEDCLIP_NO_AUG_EMBED_MEAN,
    MED_IMAGES_MEDCLIP_NO_AUG_EMBED_MEAN,
    TEXT_AUDIO_EMBED_MEAN,
    AUDIO_EMBED_MEAN,
    TEXT_TO_AUDIO_GAP_PATH
)

class GenericDataset(pl.LightningDataModule):

    def __init__(self, cfg, split='train'):
        
        print("="*80)
        print("Data split: ", split)
        print("="*80)
        
        self.split = split
        
        self.cfg = cfg
        self.dataset_type = self.cfg.data.dataset
        self.remove_modality_gap = self.cfg.data.remove_modality_gap
        self.remove_mean = self.cfg.data.remove_mean
        self.add_gaussian_noise = self.cfg.data.add_gaussian_noise
        
        data_path = self.get_data_path(cfg, split)
        
        prefix_length = cfg.model.prefix_length
        normalize_prefix = cfg.model.normalize_prefix
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        
        ###################
        print("=> Loading all_data pkl")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Number of items is %0d" % len(all_data))
        sys.stdout.flush()
        
        # {id: {x_embed: ..., y_embed: ..., y: ...}}
        self.data = all_data
        self.ids = list(self.data.keys())
        
        ###################
        
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            print("=> Loading captions_tokens, all_len dicts")
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, all_len = pickle.load(f)
        else:
            # {caption_id: tokenizer(caption)}
            print("=> Saving captions_tokens dict")
            self.captions_tokens = {sentid: 
                                torch.tensor(self.tokenizer.encode(
                                    self.data[sentid]['y']), 
                                             dtype=torch.int64) for sentid in self.ids
                               }
            print("=> Saving all_len dict")
            all_len = torch.tensor([self.captions_tokens[sentid].shape[0] for sentid in self.captions_tokens]).float()
            
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, all_len], f)
        
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
        self.output_modality = self.cfg.decoder.modality
        
        # In testing, input modality must be opposite of output modality to evaluate cross-modal task
        if self.split != "train":
            if self.output_modality == Modality.Vision:
                self.input_modality = Modality.Language
            else:
                self.input_modality = Modality.Vision
        else:
            self.input_modality = self.cfg.encoder.modality
        
        # Get all caption and image ids (they are the same)
        random.shuffle(self.ids)
        
        # Sample data
        if "train" in self.split and not OmegaConf.is_none(cfg.data, 'sample_frac'):
            sample_size = int(len(self.ids) * cfg.data.sample_frac)
            self.ids = random.sample(self.ids, sample_size)
        
        text_to_x_gap_path, text_embed_mean_path, x_embed_mean_path = self.get_gap_and_mean_paths()

        # Load modality gap
        with open(text_to_x_gap_path, 'rb') as f:
            self.text_to_x_modality_gap = pickle.load(f)
        
        # Load means gap
        with open(text_embed_mean_path, 'rb') as f:
            self.text_embed_mean = pickle.load(f)
            
        with open(x_embed_mean_path, 'rb') as f:
            self.x_embed_mean = pickle.load(f)
        
        self.std = cfg.noise_level #math.sqrt(0.016) # hyperparam from capdec paper
        
    def get_data_path(self, cfg, split):
        if split == 'train':
            data_path = cfg.data.train_data_path
        elif split == 'val':
            data_path = cfg.data.val_data_path
        elif split == 'test':
            data_path = cfg.data.test_data_path
        else:
            raise NotImplementedError(f"split {split} invalid")
            
        return data_path
    
    def get_gap_and_mean_paths(self):
        if self.dataset_type == DatasetType.Video:
            text_to_x_gap_path = TEXT_TO_VID_GAP_PATH
            text_embed_mean_path = TEXT_VIDEOCLIP_EMBED_MEAN
            x_embed_mean_path = VIDEO_EMBED_MEAN
        elif self.dataset_type == DatasetType.Medical:
            text_to_x_gap_path = TEXT_TO_MED_GAP_PATH
            # text_embed_mean_path = TEXT_MEDCLIP_EMBED_MEAN #
            # x_embed_mean_path = MED_IMAGES_MEDCLIP_EMBED_MEAN
            text_embed_mean_path = TEXT_MEDCLIP_NO_AUG_EMBED_MEAN
            x_embed_mean_path = MED_IMAGES_MEDCLIP_NO_AUG_EMBED_MEAN
        elif self.dataset_type == DatasetType.Amino_Acid:
            text_to_x_gap_path = TEXT_TO_AMINO_ACID_GAP_PATH
            text_embed_mean_path = TEXT_CLASP_EMBED_MEAN
            x_embed_mean_path = AMINO_ACID_EMBED_MEAN
        elif self.dataset_type == DatasetType.Audio:
            text_to_x_gap_path = TEXT_TO_AUDIO_GAP_PATH
            text_embed_mean_path = TEXT_AUDIO_EMBED_MEAN
            x_embed_mean_path = AUDIO_EMBED_MEAN
        else:
            raise NotImplementedError(f"dataset type {self.dataset_type} not implemented")
        
        print(text_embed_mean_path)
        print(x_embed_mean_path)
        return text_to_x_gap_path, text_embed_mean_path, x_embed_mean_path
    
    def __len__(self) -> int:
        return len(self.ids) # b/c 1:1 img2text mapping

    def pad_tokens(self, item: int):
        """
        Note: this is only for language generation 
        (the image padding is in the forward fn of ViT Decoder)
        """
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # For testing, assume cross-modal
        
        if (self.split != "train" and self.output_modality == Modality.Language) or \
            (self.input_modality == Modality.Vision and self.output_modality == Modality.Vision):
            return self.get_item_per_image(item)
        
        # Default for language generation
        id_ = self.ids[item]
        
        x_prefix = torch.from_numpy(self.data[id_]["x_embed"]).float()
        text_prefix = torch.from_numpy(self.data[id_]["y_embed"]).float()
        
        if self.normalize_prefix:
            x_prefix = x_prefix / x_prefix.norm(2, -1)
            text_prefix = text_prefix / text_prefix.norm(2, -1)
        
        if self.remove_modality_gap:
            # note: the gap was computed as vid - text
            x_prefix -= self.text_to_x_modality_gap.squeeze()
        elif self.remove_mean:
            x_prefix -= self.x_embed_mean.squeeze()
            text_prefix -= self.text_embed_mean.squeeze()
        
        if self.add_gaussian_noise:
            x_prefix += torch.randn(x_prefix.shape) * self.std
            text_prefix += torch.randn(text_prefix.shape) * self.std
            
        # Get output
        if self.output_modality == Modality.Language:
            tokens, mask = self.pad_tokens(id_)
            label = (tokens, mask)
        
        # Re-normalize
        x_prefix = torch.nn.functional.normalize(x_prefix, dim=-1)
        text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)
        return (x_prefix, text_prefix), label, id_, id_
    
    def get_item_per_image(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for iterating over images (image captioning or image reconstruction)
        id_ = self.ids[item]
        x_prefix = torch.from_numpy(self.data[id_]["x_embed"]).float()

        if self.normalize_prefix:
            x_prefix = x_prefix / x_prefix.norm(2, -1)
        
        if self.remove_modality_gap:
            # note: the gap was computed as vid - text
            x_prefix -= self.text_to_x_modality_gap.squeeze()
        elif self.remove_mean:
            x_prefix -= self.x_embed_mean.squeeze()

        dummy_prefix = torch.zeros_like(x_prefix)
        dummy_tokens = torch.zeros(self.max_seq_len)
        dummy_mask = torch.cat((torch.ones(self.prefix_length), dummy_tokens), dim=0)

        # Re-normalize
        x_prefix = torch.nn.functional.normalize(x_prefix, dim=-1)
        return (x_prefix, dummy_prefix), (dummy_tokens, dummy_mask), id_, id_
    
## To get stuff:
# video_embed = self.data[video_id]["x_embed"]
# caption_embed = self.data[video_id]["y_embed"]
# caption = self.data[video_id]["y"]
