import pytorch_lightning as pl
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer
import torch
import pandas as pd

from pytorch_lightning.loggers import WandbLogger

from .. import builder
from ..enums import Modality
from ..eval import generate2, evaluate_on_coco_caption
from ..utils import add_predictions_to_results_json, get_pred_filename, get_metrics_out_filename, evaluate_list
from ..parse_data import LABELS_JSONS_LST, get_label_json_list
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
            return {'optimizer': optimizer, 
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
            return optimizer
    
    def training_step(self, batch, batch_idx):
        self.input_modality = Modality.Language
        loss, outputs = self.shared_loss_step(batch, split='train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.cfg.val_eval:
            ## TODO: change evaluation to use the huggingface evaluate library. Also output sample captions
            self.input_modality = Modality.Vision
            return self.quick_eval_step(batch, 'val')
        else:
            if self.cfg.cross_modal_val:
                self.input_modality = Modality.Vision
            else:
                self.input_modality = Modality.Language
            loss, outputs = self.shared_loss_step(batch, split='val')
            return loss
        
    def test_step(self, batch, batch_idx):
        self.input_modality = Modality.Vision
        return self.eval_step(batch, split='test')

    def get_prefix_and_labels(self, batch):
        prefix, labels, gold_caption, img_id, cap_id = batch
        
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
            loss = self.loss(outputs, labels)
        else: # Vision
            outputs = self.model(x=prefix)
            loss = self.loss(outputs, labels, self.model)
        
        self.log(f"{split}_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss, outputs
    

    def eval_step(self, batch, split):
        # Note: batch size = 1
        _, _, gold_caption, img_id, cap_id = batch
        
        prefix, _ = self.get_prefix_and_labels(batch)
            
        prefix_embed = self.model.clip_project(prefix).reshape(-1, self.model.prefix_length, 
                                                               self.model.embed_size)
        pred = generate2(self.model, self.tokenizer, embed=prefix_embed)

        return {"image_id": img_id.item(), "caption": pred, "id": cap_id.item()}
    
    def quick_eval_step(self, batch, split):
        # Note: batch size = 1
        _, _, gold_captions, img_id, cap_id = batch
        
        prefix, _ = self.get_prefix_and_labels(batch)
        prefix_embed = self.model.clip_project(prefix).reshape(-1, self.model.prefix_length, 
                                                               self.model.embed_size)
        pred = generate2(self.model, self.tokenizer, embed=prefix_embed)

        return {"pred": pred, "refs": gold_captions}


    def validation_epoch_end(self, val_step_outputs):
        if self.cfg.val_eval:
            preds = [o['pred'] for o in val_step_outputs]
            refs = [o['refs'] for o in val_step_outputs]
            scores = evaluate_list(preds, refs)
        
            for metric, val in scores.items():
                self.log(f"val/{metric}", val, on_epoch=True, logger=True, prog_bar=True)
            
            for output in val_step_outputs[:10]:
                # TODO: test if this works!
                pred = output['pred']
                refs = output['refs']

                df = pd.DataFrame({'pred': pred, 'refs': refs})
                self.logger.log_text('generations', dataframe=df)

    def test_epoch_end(self, test_step_outputs):   
        return self.shared_epoch_end(test_step_outputs, 'test') 

    def shared_epoch_end(self, outputs, split):   
        # import pdb; pdb.set_trace() 
        # Write predictions to json
        if split == 'test':
            split = self.cfg.test_split 

        epoch = self.current_epoch if self.current_epoch else 0
        add_predictions_to_results_json(predictions=outputs, 
                                        output_dir=self.cfg.output_dir, 
                                        split=split, 
                                        epoch=epoch)
        
        # Compute eval metrics
        pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch)
        print(f"=> Predictions at {pred_file}")

        out_file = get_metrics_out_filename(self.cfg.output_dir, split, epoch=epoch)
        if self.cfg.data.dataset == 'coco':
            metrics_dict = evaluate_on_coco_caption(pred_file, LABELS_JSONS_LST[split], out_file)
        else:
            metrics_dict = evaluate_on_coco_caption(pred_file, get_label_json_list(self.cfg.data.dataset)[split], out_file)
        
        print(f"=> Metrics at {out_file}")
        
        # Log eval metrics
        for k, v in metrics_dict.items():
            self.log(f"{split}/{k}", v, on_epoch=True, logger=True, prog_bar=True)
            
    