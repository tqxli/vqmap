import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from vqmap.trainer.base import *


class TrainerEngine(EngineBase):
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
        prequant, tokens = [], []
        for batch in tqdm.tqdm(dataloader):
            batch = self._data_to_device(batch)
            z_q, z_info, z = self.model.encode(batch)
            token = z_info[-1][-1].detach().cpu().numpy()
            tokens.append(token)
            z = z.detach().cpu().permute(0, 2, 1)
            prequant.append(z.reshape(-1, z.shape[-1]).numpy())

        tokens = np.concatenate(tokens, axis=0)
        prequant = np.concatenate(prequant, axis=0)
        
        return tokens, prequant

    @torch.no_grad()
    def retrieve_latents(self, dataloader, augment=None):
        self.model.eval()
        inputs, latent_vars = [], []
        indices = []
        for batch in tqdm.tqdm(dataloader):
            # distribution, latent_vector = self.model.encode(batch)
            batch = self._data_to_device(batch)
            
            if augment == 'LR':
                motion = batch['motion']
                motion = motion.reshape(*motion.shape[:2], -1, 3)
                motion[:, :, :, 1] = -motion[:, :, :, 1]
                batch['motion'] = motion.reshape(*motion.shape[:2], -1)
            
            out = self.model.encode(batch)
            if hasattr(self.model, "quantizer"):
                try:
                    out = out[2][-1].detach().cpu().numpy()
                except:
                    out = torch.stack([o[-1].detach().cpu() for o in out[2]], -1).numpy()
            elif self.model.vae:
                out = out.permute(0, 2, 1)
                # out = out.reshape(-1, out.shape[-1])
                out = out.detach().cpu().numpy()
            else:
                out = [o[1].detach().cpu().numpy() for o in out[-1]]
            
            # latent_vars.append(latent_vector.detach().cpu())
            # inputs.append(batch[self.model.modality].detach().cpu())
            indices.append(out)
        
        # latent_vars = torch.cat(latent_vars, 0)
        # inputs = torch.cat(inputs, 0)
        indices = np.concatenate(indices, 0)
        print(indices.shape)
        return inputs, latent_vars, indices
    
    def _train_epoch(self, dataloader, cur_epoch, prefix=''):
        self.model.train()
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            dataloader = dataloader.sampling_generator(self.config.dataloader.num_samples, self.config.dataloader.batch_size)
        # generator_len = self.config.dataloader.num_samples // self.config.dataloader.batch_size
        
        tot_losses = 0
        train_losses = 0
        total_num_samples = 0
        
        for idx, batch in tqdm.tqdm(enumerate(dataloader)):
            batch = self._data_to_device(batch)
            out, loss, loss_dict = self.model(batch)

            self.optimizer.zero_grad()

            loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()
            
            tot_losses += loss.item()
            train_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1
        
        loss_names = list(loss_dict.keys())
        train_losses /= total_num_samples
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses))
        logger.info(f"Epoch: {cur_epoch} Total Loss: {tot_losses/total_num_samples} | {losses_str}")
        for name, loss in zip(loss_names, train_losses):
            self.tb_logger.add_scalar(f'train/{name}', loss, cur_epoch)