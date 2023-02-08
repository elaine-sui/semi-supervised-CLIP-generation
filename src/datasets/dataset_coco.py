import torch
import os
import pickle
import sys
import json
import random
from typing import Tuple, Optional, Union

import skimage.io as io
import clip
from PIL import Image

import pytorch_lightning as pl
from transformers import GPT2Tokenizer

from ..enums import Modality

class ClipCocoDataset(pl.LightningDataModule):

    def __init__(self, cfg, split='train'):
        
        print("="*80)
        print("Data split: ", split)
        print("="*80)
        
        self.split = split
        
        self.cfg = cfg
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
        print("Number of images is %0d" % len(all_data["images"]))
        print("Number of captions is %0d" % len(all_data["captions"]))
        sys.stdout.flush()
        
        # {image_id: {"img_path": ..., "embed": ...}}
        self.images = all_data["images"]
        # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
        self.captions = all_data["captions"]
        
        ###################
        
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            print("=> Loading caption_id_2_image_id, captions_tokens, all_len dicts")
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption_id_2_image_id, all_len = pickle.load(f)
        else:
            # {caption_id: image_id}
            print("=> Saving caption_id_2_image_id dict")
            self.caption_id_2_image_id = {sentid: self.captions[sentid]["img_id"] for sentid in self.captions}
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
                pickle.dump([self.captions_tokens, self.caption_id_2_image_id, all_len], f)
        
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
        self.output_modality = self.cfg.decoder.modality
        
        # Get all caption and image ids
        self.img_ids = sorted(list(self.images.keys()))
        random.Random(self.cfg.data.seed).shuffle(self.img_ids)
        self.cap_ids = sorted(list(self.captions.keys()))
        random.Random(self.cfg.data.seed).shuffle(self.cap_ids)
        
        ## Preprocess image to clip
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, self.preprocess_img = clip.load(self.cfg.encoder.clip_model_type, device=device, 
                                           jit=False)
        
    def get_data_path(self, cfg, split):
        if split == 'train':
            data_path = cfg.data.train_data_path
        elif split == 'val':
            data_path = cfg.data.val_data_path
        elif split == 'test':
            data_path = cfg.data.test_data_path
        elif split == 'restval':
            data_path = cfg.data.restval_data_path
        else:
            raise NotImplementedError(f"split {split} invalid")
            
        return data_path
    
    def __len__(self) -> int:
        if self.split == "test" and self.cfg.decoder.modality == Modality.Language:
            return len(self.img_ids)
        else:
            return len(self.cap_ids)

    # Note: this is only for language generation (need an image version)
    def pad_tokens(self, item: int):
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
        if self.split == "test" and self.cfg.encoder.modality == Modality.Vision \
                and self.cfg.decoder.modality == Modality.Language:
            return self.get_item_for_test_img_cap(item)
        
        # TODO: Item is a sentence (caption) id -- need to change for image generation
        item = self.cap_ids[item]
        img_id = self.caption_id_2_image_id[item]
        
        img_prefix = self.images[img_id]["embed"]
        text_prefix = self.captions[item]["embed"]
        
        if self.normalize_prefix:
            img_prefix = img_prefix.float()
            img_prefix = img_prefix / img_prefix.norm(2, -1)
            
            text_prefix = text_prefix.float()
            text_prefix = text_prefix / text_prefix.norm(2, -1)
            
        # Get output
        if self.output_modality == Modality.Vision:
            img_path = self.images[img_id]["img_path"]
            image = io.imread(img_path)
            label = self.preprocess_img(Image.fromarray(image)).unsqueeze(0)
        else:
            tokens, mask = self.pad_tokens(item)
            label = (tokens, mask)
        
        return (img_prefix, text_prefix), label, img_id, item
    
    def get_item_for_test_img_cap(self, item: int) -> Tuple[torch.Tensor, ...]:
        # image captioning -- this is so that we don't iterate over all the captions have generate
        # captions for the same image 5 times
        img_id = self.img_ids[item]
        img_prefix = self.images[img_id]["embed"]

        if self.normalize_prefix:
            img_prefix = img_prefix.float()
            img_prefix = img_prefix / img_prefix.norm(2, -1)

        dummy_prefix = torch.zeros_like(img_prefix)
        dummy_tokens, dummy_mask = self.pad_tokens(self.cap_ids[0])

        return (img_prefix, dummy_prefix), (dummy_tokens, dummy_mask), img_id, 0
            
    
## To get stuff:
# image_path = self.images[img_id]["img_path"]
# image_embed = self.images[img_id]["embed"]
# caption = self.captions[sent_id]["caption"]
# image_id_for_caption = self.captions[sent_id]["img_id"]
# caption_embed = self.captions[sent_id]["embed"]
