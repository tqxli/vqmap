"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import hashlib
import datetime
from loguru import logger
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from vqmap.models import initialize_model
from vqmap.optimizers import get_optimizer, get_lr_scheduler
from vqmap.utils.serialize import torch_safe_load, flatten_dict
from vqmap.datasets.augmentation import *


class EngineBase(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.evaluator = None

        self.config = None

        self.metadata = {}

    def create(self, config, verbose=False):
        self.config = config
        self.n_epochs = config.train.epochs
        self.set_model(initialize_model(config.model))
        logger.info('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        params = [param for param in self.model.parameters()
                  if param.requires_grad]

        self.set_optimizer(get_optimizer(config.optimizer.name,
                                         params,
                                         config.optimizer))
        self.set_lr_scheduler(get_lr_scheduler(config.lr_scheduler.name,
                                               self.optimizer,
                                               config.lr_scheduler,
                                               logger))

        logger.info('Engine is created.')
        logger.info(config)

        # logger.update_tracker({'full_config': config}, keys=['full_config'])

    def set_model(self, model):
        self.model = model

    def model_to_device(self):
        self.model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_logger(self, logger):
        logger = logger
    
    def set_tb(self, tb_logger):
        self.tb_logger = tb_logger

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

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
        logger.info('Start train ...')

        self.model_to_device()
        if self.config.train.get('use_fp16'):
            logger.info('Train with half precision')
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
            logger.info('expected remain {}'.format(expected_remain))
        logger.info('finish train, takes {}'.format(datetime.datetime.now() - dt))

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if 'lr' in metadata:
            report_dict['{}lr'.format(prefix)] = metadata['lr']

        logger.report(report_dict,
                           prefix='[Eval] Report @step: ',
                           pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        logger.update_tracker(tracker_data)

    @torch.no_grad()
    def evaluate(self, dataloader, cur_epoch):
        self.model.eval()
        
        tot_losses = 0
        valid_losses = 0
        total_num_samples = 0
        
        for idx, batch in enumerate(dataloader):
            batch = self._batch_augmentation(batch)
            batch = self._data_to_device(batch)
            out, loss, loss_dict = self.model(batch)
            tot_losses += loss.item()
            valid_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1
        
        loss_names = list(loss_dict.keys())
        valid_losses /= total_num_samples
        losses_str = ' '.join('{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, valid_losses))
        logger.info(
            f"Epoch[{cur_epoch+1}/{self.n_epochs}]"
            + " Total Loss: {:.4f} | {}".format(tot_losses/total_num_samples, losses_str)
        )
        for name, loss in zip(loss_names, valid_losses):
            self.tb_logger.add_scalar(f'val/{name}', loss, cur_epoch+1)
        return valid_losses[0]

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': self.config,
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)

    def load_models(self, state_dict_path, load_keys=None):
        with open(state_dict_path, 'rb') as fin:
            model_hash = hashlib.sha1(fin.read()).hexdigest()
            self.metadata['pretrain_hash'] = model_hash

        state_dict = torch.load(state_dict_path, map_location='cpu')

        if 'model' not in state_dict:
            torch_safe_load(self.model, state_dict, strict=False)
            return

        if not load_keys:
            load_keys = ['model', 'optimizer', 'lr_scheduler']
        for key in load_keys:
            try:
                torch_safe_load(getattr(self, key), state_dict[key])
            except RuntimeError as e:
                logger.info('Unable to import state_dict, missing keys are found. {}'.format(e))
                torch_safe_load(getattr(self, key), state_dict[key], strict=False)
        logger.info('state dict is loaded from {} (hash: {}), load_key ({})'.format(state_dict_path,
                                                                                        model_hash,
                                                                                        load_keys))

    def load_state_dict(self, state_dict_path, load_keys=None):
        # state_dict = torch.load(state_dict_path)
        self.load_models(state_dict_path, load_keys)

    def _data_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        return batch
    
    def _batch_augmentation(self, batch):
        # [B, T, D]
        if not self.config.train.get("lr_augmentation", False):
            return batch
        
        # if np.random.random() < 0.5:
        #     return batch

        motion = batch['motion']
        motion_aug = lateral_flip(motion)
        motion_aug = torch.cat((motion, motion_aug), 0)
        batch['motion'] = motion_aug
        batch['ref'] = motion_aug.clone()
        batch['lr_augmentation'] = True
        return batch

def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']