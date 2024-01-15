import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from vqmap.trainer.base import *
    

class TrainerEngineGPT(EngineBase):
    def _train_epoch(self, dataloader, cur_epoch, prefix=''):
        self.model.train()
        
        tot_losses = 0
        train_losses = 0
        total_num_samples = 0
        right_num = 0
        total_num = 0
        
        pbar = tqdm.tqdm(dataloader)
        for batch in pbar:
            tokens, tokens_len = batch[0].to(self.device), batch[1].to(self.device)
            if len(tokens.shape) == 3:
                inputs, targets = tokens[:, 0], tokens[:, 1]
                social = True
            else:
                inputs, targets = tokens[:, :-1], tokens[:, 1:]
                social = False

            logits = self.model(inputs)[0] #[B, T, code_s]

            loss = 0.0
            for bid in range(tokens.shape[0]):
                valid_len = tokens_len[bid]
                if not social:
                    valid_len -= 1
                loss += F.cross_entropy(
                    logits[bid].view(-1, logits.shape[-1]), targets[bid].view(-1)
                )                
                
                # Accuracy
                probs = torch.softmax(logits[bid][:valid_len], dim=-1)
                _, cls_pred_index = torch.max(probs, dim=-1)
                right_num += (cls_pred_index.flatten(0) == targets[bid][:valid_len].view(-1)).sum().item()
                total_num += cls_pred_index.flatten(0).shape[0]

            loss /= tokens.shape[0]
            self.optimizer.zero_grad()

            loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
            
            tot_losses += loss.item()
            train_losses += np.array([loss.item()])
            total_num_samples += 1
        
        loss_names = ['CrossEntropy']
        train_losses /= total_num_samples
        train_acc = right_num / total_num
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses))
        logger.info(
            f"Epoch[{cur_epoch}/{self.n_epochs}] {losses_str}"
            + " | Accuracy {:.4f}%".format(100*train_acc)
        )
        for name, loss in zip(loss_names, train_losses):
            self.tb_logger.add_scalar(f'train/{name}', loss, cur_epoch)
        self.tb_logger.add_scalar(f"train/Accuracy", train_acc, cur_epoch)

    @torch.no_grad()
    def evaluate(self, dataloader, cur_epoch):
        self.model.eval()
        
        tot_losses = 0
        valid_losses = 0
        total_num_samples = 0
        total_num_samples = 0
        right_num = 0
        total_num = 0
        
        pbar = tqdm.tqdm(dataloader)
        for batch in pbar:
            tokens, tokens_len = batch[0].to(self.device), batch[1].to(self.device)
            if len(tokens.shape) == 3:
                inputs, targets = tokens[:, 0], tokens[:, 1]
                social = True
            else:
                inputs, targets = tokens[:, :-1], tokens[:, 1:]
                social = False

            logits = self.model(inputs)[0]
            
            loss = 0.0
            for bid in range(tokens.shape[0]):
                valid_len = tokens_len[bid]
                if not social:
                    valid_len -= 1
                loss += F.cross_entropy(
                    logits[bid].view(-1, logits.shape[-1]), targets[bid].view(-1)
                )
                
                # Accuracy
                probs = torch.softmax(logits[bid][:valid_len], dim=-1)
                _, cls_pred_index = torch.max(probs, dim=-1)
                right_num += (cls_pred_index.flatten(0) == targets[bid][:(valid_len)].view(-1)).sum().item()
                total_num += cls_pred_index.flatten(0).shape[0]
            
            loss = loss / tokens.shape[0]
            
            tot_losses += loss.item()
            valid_losses += np.array([loss.item()])
            total_num_samples += 1
        
        loss_names = ["CrossEntropy"]
        valid_losses /= total_num_samples
        valid_acc = right_num / total_num
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, valid_losses))
        logger.info(
            f"Epoch: {cur_epoch} Valid loss {losses_str}"
            + " | Accuracy {:.4f}%".format(100*valid_acc)
        )
        for name, loss in zip(loss_names, valid_losses):
            self.tb_logger.add_scalar(f'val/{name}', loss, cur_epoch)
        self.tb_logger.add_scalar(f"val/Accuracy", valid_acc, cur_epoch)
        
        return valid_losses[0]
