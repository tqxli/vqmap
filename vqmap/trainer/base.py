"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import hashlib
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from vqmap.models import get_model, initialize_model
from vqmap.optimizers import get_optimizer, get_lr_scheduler

from vqmap.utils.serialize import torch_safe_load
from loguru import logger

class EngineBase(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.evaluator = None

        self.config = None
        # self.logger = None

        self.metadata = {}

    def create(self, config, verbose=False):
        self.config = config
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

    @torch.no_grad()
    def evaluate(self, val_loaders, n_crossfolds=None, **kwargs):
        if self.evaluator is None:
            logger.info('[Evaluate] Warning, no evaluator is defined. Skip evaluation')
            return

        self.model_to_device()
        self.model.eval()

        if not isinstance(val_loaders, dict):
            val_loaders = {'te': val_loaders}

        scores = {}
        for key, data_loader in val_loaders.items():
            logger.info('Evaluating {}...'.format(key))
            _n_crossfolds = -1 if key == 'val' else n_crossfolds
            scores[key] = self.evaluator.evaluate(data_loader, n_crossfolds=_n_crossfolds,
                                                  key=key, **kwargs)
        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': self.config,
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)
        # logger.info('state dict is saved to {}, metadata: {}'.format(
            # save_to, json.dumps(metadata, indent=4)))

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