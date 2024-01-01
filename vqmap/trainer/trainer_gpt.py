import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from vqmap.trainer.base import EngineBase
from vqmap.utils.serialize import flatten_dict


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

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
        self.logger.log(
            f"Epoch: {cur_epoch} {losses_str}"
            + f" | Accuracy {100*train_acc}%"
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
        self.logger.log(
            f"Epoch: {cur_epoch} Valid loss {losses_str}"
            + f" | Accuracy {100*valid_acc}%"
        )
        for name, loss in zip(loss_names, valid_losses):
            self.tb_logger.add_scalar(f'val/{name}', loss, cur_epoch)
        self.tb_logger.add_scalar(f"val/Accuracy", valid_acc, cur_epoch)
        
        return valid_losses[0]

    def train(self, tr_loader, n_epochs,
              val_loaders=None,
              val_epochs=1,
              model_save_to='last.pth',
              best_model_save_to='best.pth'):

        if val_loaders and 'val' not in val_loaders:
            raise KeyError('val_loaders should contain key "val", '
                           'but ({})'.format(val_loaders.keys()))

        dt = datetime.datetime.now()

        prefix = 'train__'
        eval_prefix = ''
        self.logger.log('start train')

        self.model_to_device()
        if self.config.train.get('use_fp16'):
            self.logger.log('Train with half precision')
            self.to_half()

        best_score = 0
        for cur_epoch in range(n_epochs):
            self._train_epoch(tr_loader, cur_epoch, prefix=prefix)

            metadata = self.metadata.copy()
            metadata['cur_epoch'] = cur_epoch + 1
            metadata['lr'] = get_lr(self.optimizer)

            if val_loaders is not None and (cur_epoch + 1) % val_epochs == 0:
                scores = self.evaluate(val_loaders['val'], cur_epoch)

            if self.config.lr_scheduler.name == 'reduce_lr_on_plateau':
                self.lr_scheduler.step(scores)
            else:
                self.lr_scheduler.step()

            self.save_models(model_save_to, metadata)

            elasped = datetime.datetime.now() - dt
            expected_total = elasped / (cur_epoch + 1) * n_epochs
            expected_remain = expected_total - elasped
            self.logger.log('expected remain {}'.format(expected_remain))
        self.logger.log('finish train, takes {}'.format(datetime.datetime.now() - dt))
