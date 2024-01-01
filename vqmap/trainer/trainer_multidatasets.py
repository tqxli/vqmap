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


class TrainerEngine(EngineBase):
    def _data_to_device(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        return batch

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
        self.logger.log(f"Epoch: {cur_epoch} Valid loss {tot_losses/total_num_samples} | {losses_str}")
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
        self.logger.log(f"Epoch: {cur_epoch} Total Loss: {tot_losses/total_num_samples} | {losses_str}")
        for name, loss in zip(loss_names, train_losses):
            self.tb_logger.add_scalar(f'train/{name}', loss, cur_epoch)

    def train(self, tr_loader, n_epochs,
              val_loaders=None,
              val_epochs=1,
              model_save_to='model_last.pth',
              best_model_save_to='model_best.pth'):

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
                
                metadata['scores'] = scores

                if best_score < scores:
                    self.save_models(best_model_save_to, metadata)
                    best_score = scores
                    metadata['best_score'] = best_score
                    metadata['best_epoch'] = cur_epoch + 1

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

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if 'lr' in metadata:
            report_dict['{}lr'.format(prefix)] = metadata['lr']

        self.logger.report(report_dict,
                           prefix='[Eval] Report @step: ',
                           pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        self.logger.update_tracker(tracker_data)