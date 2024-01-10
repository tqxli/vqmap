import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from vqmap.trainer.base import *


class TrainerEngineCoembed(EngineBase):
    @torch.no_grad()
    def evaluate(self, dataloader, cur_epoch):
        self.model.eval()
        
        tot_losses = 0
        valid_losses = 0
        total_num_samples = 0
        
        for (batch0, batch1) in tqdm.tqdm(zip(dataloader[0], dataloader[1])):
            batch0 = self._data_to_device(batch0)
            batch1 = self._data_to_device(batch1)
            out, loss, loss_dict = self.model([batch0, batch1])
            tot_losses += loss.item()
            valid_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1
        
        loss_names = list(loss_dict.keys())
        valid_losses /= total_num_samples
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, valid_losses))
        logger.info(f"Epoch: {cur_epoch} Valid loss {tot_losses/total_num_samples} | {losses_str}")
        for name, loss in zip(loss_names, valid_losses):
            self.tb_logger.add_scalar(f'val/{name}', loss, cur_epoch)
        return valid_losses[0]

    @torch.no_grad()
    def inference(self, dataloader):
        self.model.eval()
        outputs = []
        inputs = []
        for batch in tqdm.tqdm(dataloader):
            # distribution, latent_vector = self.model.encode(batch)
            batch = self._data_to_device(batch)
            out = self.model(batch)[0].detach().cpu()
            # out = out.reshape(out.shape[0]*out.shape[1], -1)
            target = batch['motion'].detach().cpu()#.reshape(*out.shape)
            # loss = ((out-target)**2).sum(-1).sqrt().mean(1)
            outputs.append(out)
            inputs.append(target)
            # losses.append(loss)
        
        outputs = torch.cat(outputs, 0).numpy()
        inputs = torch.cat(inputs, 0).numpy()
        # losses = torch.cat(losses, 0).numpy()
        return outputs, inputs

    @torch.no_grad()
    def retrieve_vq(self, dataloader):
        self.model.eval()
        vq_results = []
        for batch in tqdm.tqdm(dataloader):
            batch = self._data_to_device(batch)
            latent, emb_loss, info = self.model.encode(batch)
            if hasattr(self.model, "quantizer"):
                info = [i.detach().cpu().numpy() for i in info]
            vq_results.append(info)
        return vq_results

    @torch.no_grad()
    def retrieve_latents(self, dataloader):
        self.model.eval()
        inputs, latent_vars = [], []
        indices = []
        for batch in tqdm.tqdm(dataloader):
            # distribution, latent_vector = self.model.encode(batch)
            batch = self._data_to_device(batch)
            out = self.model.encode(batch)
            
            if hasattr(self.model, "quantizer"):
                try:
                    out = out[-1][-1].detach().cpu().numpy()
                except:
                    out = torch.stack([o[-1].detach().cpu() for o in out[-1]], -1).numpy()
            else:
                out = [o[1].detach().cpu().numpy() for o in out[-1]]
            
            # latent_vars.append(latent_vector.detach().cpu())
            # inputs.append(batch[self.model.modality].detach().cpu())
            indices.append(out)
        
        # latent_vars = torch.cat(latent_vars, 0)
        # inputs = torch.cat(inputs, 0)
        # indices = torch.cat(indices, 0)
        
        return inputs, latent_vars, indices
    
    def _train_epoch(self, dataloader, cur_epoch, prefix=''):
        self.model.train()

        tot_losses = 0
        train_losses = 0
        total_num_samples = 0
        
        for (batch0, batch1) in tqdm.tqdm(zip(dataloader[0], dataloader[1])):
            batch0 = self._data_to_device(batch0)
            batch1 = self._data_to_device(batch1)
            out, loss, loss_dict = self.model([batch0, batch1])

            self.optimizer.zero_grad()

            loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
            
            tot_losses += loss.item()
            train_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1

            # if (idx + 1) % self.config.train.log_step == 0:
            #     loss_dict = {'{}{}'.format(prefix, key): val
            #                  for key, val in loss_dict.items()}
            #     loss_dict['step'] = cur_step(cur_epoch, idx, generator_len)
            #     self.logger.report(loss_dict,
            #                        prefix='[Train] Report @step: ')
        
        loss_names = list(loss_dict.keys())
        train_losses /= total_num_samples
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses))
        logger.info(f"Epoch: {cur_epoch} Total Loss: {tot_losses/total_num_samples} | {losses_str}")
        for name, loss in zip(loss_names, train_losses):
            self.tb_logger.add_scalar(f'train/{name}', loss, cur_epoch)
