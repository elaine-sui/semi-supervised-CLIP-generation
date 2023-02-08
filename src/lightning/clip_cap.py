import json
import pytorch_lightning as pl
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer

from .. import builder
from ..enums import Modality
from ..eval import generate2, evaluate_on_coco_caption
from ..utils import add_predictions_to_results_json, get_pred_filename, get_metrics_out_filename
from ..parse_data import LABELS_JSONS_LST
from ..models import Decoder

class ClipCaptionLightningModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = Decoder(cfg)
        self.loss = builder.build_loss(cfg)
        self.prefix_length = self.model.prefix_length
        self.cfg = cfg
        
        self.input_modality = cfg.encoder.modality
        self.output_modality = cfg.decoder.modality
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        
        if not OmegaConf.is_none(self.cfg.train, "scheduler"):
            scheduler = builder.build_scheduler(self.cfg, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
    
    def training_step(self, batch, batch_idx):       
        return self.shared_loss_step(batch, split='train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_loss_step(batch, split='val')
        
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, split='test')

    def get_prefix_and_labels(self, batch):
        prefix, labels, img_id, cap_id = batch
        if self.input_modality == Modality.Vision:
            prefix = prefix[0]
        elif self.input_modality == Modality.Language:
            prefix = prefix[1]
        else: # both
            prefix = torch.cat(prefix, dim=0)
            if self.output_modality == Modality.Language:
                if labels is not None:
                    labels = (labels[0].repeat((2, 1)), labels[1].repeat((2, 1)))
                
        return prefix, labels
        
    
    def shared_loss_step(self, batch, split):
        prefix, labels = self.get_prefix_and_labels(batch)
        
        if self.output_modality == Modality.Language:
            (labels, mask) = labels
            outputs = self.model(tokens=labels, prefix=prefix, mask=mask)
            outputs = outputs.logits[:, self.prefix_length - 1: -1]
        else: # Vision
            outputs = self.model(x=prefix)
        
        loss = self.loss(outputs, labels)
        self.log(f"{split}_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    
    def eval_step(self, batch, split):
        # Note: batch size = 1
        _, _, img_id, cap_id = batch
        
        prefix, _ = self.get_prefix_and_labels(batch)
            
        prefix_embed = self.model.clip_project(prefix).reshape(-1, self.model.prefix_length, 
                                                               self.model.embed_size)
        pred = generate2(self.model, self.tokenizer, embed=prefix_embed)

        return {"image_id": img_id.item(), "caption": pred, "id": cap_id.item()}

    def test_epoch_end(self, test_step_outputs):        
        # Write predictions to json
        split="test"
        add_predictions_to_results_json(predictions=test_step_outputs, 
                                        output_dir=self.cfg.output_dir, split=split)
        
        # Compute eval metrics
        pred_file = get_pred_filename(self.cfg.output_dir, split)
        out_file = get_metrics_out_filename(self.cfg.output_dir, split)
        metrics_dict = evaluate_on_coco_caption(pred_file, LABELS_JSONS_LST[split], out_file)
        
        print(f"=> Metrics at {out_file}")
        
        # Log eval metrics
        for k, v in metrics_dict.items():
            self.log(f"{split}/{k}", v, on_epoch=True, logger=True, prog_bar=True)