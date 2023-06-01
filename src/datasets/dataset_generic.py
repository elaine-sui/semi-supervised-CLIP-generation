import torch
import pickle
import sys
import random

from typing import Tuple

import pytorch_lightning as pl
from transformers import GPT2Tokenizer
from omegaconf import OmegaConf

from ..enums import Modality, DatasetType
from ..parse_data import (
    TEXT_TO_VID_GAP_PATH, 
    TEXT_AUDIO_EMBED_MEAN,
    AUDIO_EMBED_MEAN,
    TEXT_TO_AUDIO_GAP_PATH,
    TEXT_IMAGEBIND_VIDEO_EMBED_MEAN,
    VIDEO_IMAGEBIND_EMBED_MEAN
)

# TODO Re-do data splits and whatnot with audio and video b/c of 1-to-X mapping!!!!

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
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.gpt2_type)
        self.prefix_length = cfg.model.prefix_length
        self.normalize_prefix = cfg.model.normalize_prefix
        self.re_normalize_prefix = cfg.model.re_normalize_prefix
        
        ###################
        print("=> Loading all_data pkl")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Number of items is %0d" % len(all_data))
        sys.stdout.flush()
        
        # {input_id: {"img_path": ..., "embed": ...}}
        self.inputs = all_data["inputs"]
        # {caption_id: {"caption": .., "input_id": .., "embed": ...}}
        self.captions = all_data["captions"]
        
        ###################
        
        # {caption_id: image_id}
        print("=> Saving caption_id_2_input_id dict")
        self.caption_id_2_input_id = {sentid: self.captions[sentid]["input_id"] for sentid in self.captions}

        print("=> Saving input_id_2_caption_ids dict")
        self.input_id_2_caption_ids = {}
        for sentid in self.captions:
            input_id = self.captions[sentid]['input_id']
            if input_id not in self.input_id_2_caption_ids:
                self.input_id_2_caption_ids[input_id] = []
            self.input_id_2_caption_ids[input_id].append(sentid)

        # {caption_id: tokenizer(caption)}
        print("=> Saving captions_tokens dict")
        self.captions_tokens = {sentid: 
                            torch.tensor(self.tokenizer.encode(
                                self.captions[sentid]['caption']), 
                                            dtype=torch.int64) for sentid in self.captions
                            }
        print("=> Saving all_len dict")
        all_len = torch.tensor([self.captions_tokens[sentid].shape[0] for sentid in self.captions_tokens]).float()
        
        with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
            pickle.dump([self.captions_tokens, self.caption_id_2_input_id, self.input_id_2_caption_ids, all_len], f)
        
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
        self.output_modality = self.cfg.decoder.modality
        
        # In testing, input modality must be opposite of output modality to evaluate cross-modal task
        if self.cfg.cross_modal_val:
            self.condition = self.split != "train"
        else:
            self.condition = self.split == "test"
        
        if self.condition:
            if self.output_modality == Modality.Vision:
                self.input_modality = Modality.Language
            else:
                self.input_modality = Modality.Vision
        else:
            self.input_modality = self.cfg.encoder.modality
        
        # Get all caption and image ids
        self.input_ids = sorted(list(self.inputs.keys()))
        random.shuffle(self.input_ids)
        self.cap_ids = sorted(list(self.captions.keys()))
        random.shuffle(self.cap_ids)
        
        # Sample data
        if "train" in self.split and not OmegaConf.is_none(cfg.data, 'sample_frac'):
            input_sample_size = int(len(self.input_ids) * cfg.data.sample_frac)
            cap_sample_size = int(len(self.cap_ids) * cfg.data.sample_frac)
            self.input_ids = random.sample(self.input_ids, input_sample_size)
            self.cap_ids = random.sample(self.cap_ids, cap_sample_size)
        
        print(f"Input dataset size (after possibly subsampling): {len(self.input_ids)}")
        print(f"Caption dataset size (after possibly subsampling): {len(self.cap_ids)}")

        text_to_x_gap_path, text_embed_mean_path, x_embed_mean_path = self.get_gap_and_mean_paths()

        # Load modality gap
        with open(text_to_x_gap_path, 'rb') as f:
            self.text_to_x_modality_gap = pickle.load(f)
        
        # Load means gap
        with open(text_embed_mean_path, 'rb') as f:
            self.text_embed_mean = pickle.load(f)
            
        with open(x_embed_mean_path, 'rb') as f:
            self.x_embed_mean = pickle.load(f)
        
        self.std = cfg.noise_level
        
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
            text_embed_mean_path = TEXT_IMAGEBIND_VIDEO_EMBED_MEAN
            x_embed_mean_path = VIDEO_IMAGEBIND_EMBED_MEAN
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
        if (self.condition and self.output_modality == Modality.Language):
            # Image captioning testing
            return len(self.input_ids)
        elif (self.input_modality == Modality.Vision and \
              self.output_modality == Modality.Vision):
            # Image reconstruction
            return len(self.input_ids)
        else:
            return len(self.cap_ids)

    def pad_tokens(self, item: int):
        """
        Note: this is only for language generation 
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
        
        if (self.condition and self.output_modality == Modality.Language) or \
            (self.input_modality == Modality.Vision and self.output_modality == Modality.Vision):
            return self.get_item_per_image(item)
        
        # Default for language generation
        item = self.cap_ids[item]
        input_id = self.caption_id_2_input_id[item]
        
        x_prefix = self.inputs[input_id]["embed"].float()
        text_prefix = self.captions[item]["embed"].float()
        text = self.captions[item]['caption']
        
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
            tokens, mask = self.pad_tokens(item)
            label = (tokens, mask)
        
        # Re-normalize
        if self.re_normalize_prefix:
            x_prefix = torch.nn.functional.normalize(x_prefix, dim=-1)
            text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)
        return (x_prefix, text_prefix), label, [text], input_id, item
    
    def get_item_per_image(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for iterating over images (image captioning or image reconstruction)
        input_id = self.input_ids[item]
        x_prefix = self.inputs[input_id]["embed"].float()

        if self.normalize_prefix:
            x_prefix = x_prefix / x_prefix.norm(2, -1)
        
        if self.remove_modality_gap:
            # note: the gap was computed as vid - text
            x_prefix -= self.text_to_x_modality_gap.squeeze()
        elif self.remove_mean:
            x_prefix -= self.x_embed_mean.squeeze()

        # get all reference captions for input
        caption_ids = self.input_id_2_caption_ids[input_id]
        text = [self.captions[cap_id]['caption'] for cap_id in caption_ids]

        dummy_prefix = torch.zeros_like(x_prefix)
        dummy_tokens = torch.zeros(self.max_seq_len)
        dummy_mask = torch.cat((torch.ones(self.prefix_length), dummy_tokens), dim=0)

        # Re-normalize
        if self.re_normalize_prefix:
            x_prefix = torch.nn.functional.normalize(x_prefix, dim=-1)
        return (x_prefix, dummy_prefix), (dummy_tokens, dummy_mask), text, input_id, item
    
## To get stuff:
# file_path = self.inputs[input_id]["file_path"]
# input_embed = self.inputs[input_id]["embed"]
# caption = self.captions[sent_id]["caption"]
# image_id_for_caption = self.captions[sent_id]["img_id"]
# caption_embed = self.captions[sent_id]["embed"]
