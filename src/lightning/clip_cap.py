import pytorch_lightning as pl
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import torch
import pandas as pd
# import torch.distributed as dist
import json
import pickle

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
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.decoder.model) 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({"pad_token":"<pad>"})

        self.model = Decoder(cfg, self.tokenizer)
        self.loss = builder.build_loss(cfg)
        self.prefix_length = self.model.prefix_length
        self.cfg = cfg
        
        self.input_modality = cfg.encoder.modality
        self.output_modality = cfg.decoder.modality

        self.validation_step_outputs = []
        self.test_step_outputs = []
    
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
            self.input_modality = Modality.Vision
            out = self.quick_eval_step(batch, 'val')
            self.validation_step_outputs.append(out)
        else:
            if self.cfg.cross_modal_val:
                self.input_modality = Modality.Vision
            else:
                self.input_modality = Modality.Language
            loss, outputs = self.shared_loss_step(batch, split='val')
            return loss
        
    def test_step(self, batch, batch_idx):
        self.input_modality = Modality.Vision
        out = self.eval_step(batch, split='test')
        self.test_step_outputs.append(out)

    def get_prefix_and_labels(self, batch):
        prefix, labels, gold_caption, img_id, cap_id = batch

        return prefix, labels
        
        # if self.input_modality == Modality.Vision:
        #     prefix = prefix[0]
        # elif self.input_modality == Modality.Language:
        #     prefix = prefix[1]
        # else: # both
        #     prefix = torch.cat(prefix, dim=0)
        #     if self.output_modality == Modality.Language:
        #         if labels is not None:
        #             labels = (labels[0].repeat((2, 1)), labels[1].repeat((2, 1)))
                
        # return prefix, labels
        
    
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


    def on_validation_epoch_end(self):
        if self.cfg.val_eval and len(self.validation_step_outputs) > 0:
            preds = [o['pred'] for o in self.validation_step_outputs]
            refs = [o['refs'] for o in self.validation_step_outputs]

            # Save to files
            epoch = self.current_epoch if self.current_epoch else 0
            pred_file = get_pred_filename(self.cfg.output_dir, split='val', epoch=epoch, rank=self.local_rank,)

            with open(pred_file, 'wb') as f:
                pickle.dump({'preds': preds, 'refs': refs}, f)

            print(f"=> Predictions at {pred_file}")

            # dist.barrier()

            if self.local_rank == 0:
                # Read from files and combine
                all_preds, all_refs = preds, refs
                # all_preds, all_refs = [], []
                # for i in [0]: #range(dist.get_world_size()):
                #     pred_file = get_pred_filename(self.cfg.output_dir, split='val', epoch=epoch, rank=self.local_rank, temp=True)

                #     with open(pred_file, 'rb') as f:
                #         rank_data = pickle.load(f)
                #         all_preds.extend(rank_data['preds'])
                #         all_refs.extend(rank_data['refs'])

                scores = evaluate_list(all_preds, all_refs)
            
                for metric, val in scores.items():
                    self.log(f"val/{metric}", val, on_epoch=True, logger=True, prog_bar=True)
                
                if isinstance(self.logger, WandbLogger):
                    for i, output in enumerate(self.validation_step_outputs[:10]):
                        pred = output['pred']
                        refs = ['\n'.join(r) for r in output['refs']]

                        df = pd.DataFrame({'pred': pred, 'refs': refs})
                        self.logger.log_text(key=f'generated_caption_{i}', dataframe=df)
            self.validation_step_outputs.clear()

            # dist.barrier()

    def on_test_epoch_end(self):   
        self.shared_epoch_end(self.test_step_outputs, 'test') 
        self.test_step_outputs.clear()
    
    def shared_epoch_end(self, outputs, split):   
        # Write predictions to json
        if split == 'test':
            split = self.cfg.test_split 

        epoch = self.current_epoch if self.current_epoch else 0
        
        # Compute eval metrics
        pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch)
        
        add_predictions_to_results_json(predictions=outputs, 
                                        filepath=pred_file)
        
        print(f"=> Predictions at {pred_file}")

        out_file = get_metrics_out_filename(self.cfg.output_dir, split, epoch=epoch)
        if self.cfg.data.dataset == 'coco':
            metrics_dict = evaluate_on_coco_caption(pred_file, LABELS_JSONS_LST[split], out_file)

        print(f"=> Metrics at {out_file}")
        
        # Log eval metrics
        for k, v in metrics_dict.items():
            self.log(f"{split}/{k}", v, on_epoch=True, logger=True, prog_bar=True)


    # def shared_epoch_end(self, outputs, split):   
    #     # import pdb; pdb.set_trace() 
    #     # Write predictions to json
    #     if split == 'test':
    #         split = self.cfg.test_split 

    #     epoch = self.current_epoch if self.current_epoch else 0
        
    #     # Compute eval metrics
    #     pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch, rank=self.local_rank)
        
    #     add_predictions_to_results_json(predictions=outputs, 
    #                                     filepath=pred_file)
        
    #     print(f"=> Predictions at {pred_file}")

    #     # dist.barrier()

    #     if self.local_rank == 0: # combine all the predictions in a single pred file
    #         all_preds = []
    #         for i in [0]: #range(dist.get_world_size()):
    #             pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch, rank=i)

    #             with open(pred_file, 'r') as f:
    #                 all_preds.extend(json.load(f))
            
    #         pred_file = get_pred_filename(self.cfg.output_dir, split, epoch=epoch, rank='full')

    #         add_predictions_to_results_json(predictions=all_preds, 
    #                                     filepath=pred_file)

    #         out_file = get_metrics_out_filename(self.cfg.output_dir, split, epoch=epoch)
    #         if self.cfg.data.dataset == 'coco':
    #             metrics_dict = evaluate_on_coco_caption(pred_file, LABELS_JSONS_LST[split], out_file)
    #         else:
    #             metrics_dict = evaluate_on_coco_caption(pred_file, get_label_json_list(self.cfg.data.dataset)[split], out_file)
            
    #         print(f"=> Metrics at {out_file}")
            
    #         # Log eval metrics
    #         for k, v in metrics_dict.items():
    #             self.log(f"{split}/{k}", v, on_epoch=True, logger=True, prog_bar=True)

    #     # dist.barrier()
            
    