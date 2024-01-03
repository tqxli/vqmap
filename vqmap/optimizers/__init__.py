"""
PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import warnings
import torch.optim as optim
from loguru import logger


def get_optimizer(optimizer_name, parameters, config):
    if logger:
        logger.info('creating [{}] from Config({})'.format(optimizer_name, config))
    if optimizer_name == 'adam':
        if set(config.keys()) - {'learning_rate', 'betas', 'eps',
                                 'weight_decay', 'amsgrad', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        optimizer = optim.Adam(parameters,
                               lr=config.learning_rate,
                               betas=config.get('betas', (0.9, 0.999)),
                               eps=float(config.get('eps', 1e-8)),
                               weight_decay=float(config.get('weight_decay', 0)),
                               amsgrad=config.get('amsgrad', False))
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(parameters, config.learning_rate)
    else:
        raise ValueError(f'Invalid optimizer name: {optimizer_name}')
    return optimizer


def get_lr_scheduler(scheduler_name, optimizer, config, logger=None):
    if scheduler_name == 'reduce_lr_on_plateau':
        if set(config.keys()) - {'mode', 'factor', 'patience',
                                 'verbose', 'threshold',
                                 'threshold_mode', 'cooldown',
                                 'min_lr', 'eps', 'name'}:
            warnings.warn('found unused keys in {}'.format(config.keys()))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=config.get('mode', 'min'),
            factor=float(config.get('factor', 0.1)),
            patience=config.get('patience', 10),
            verbose=config.get('verbose', True),
            threshold=float(config.get('threshold', 1e-4)),
            threshold_mode=config.get('threshold_mode', 'rel'),
            cooldown=float(config.get('cooldown', 0)),
            min_lr=float(config.get('min_lr', 0)),
            eps=float(config.get('eps', 1e-8)))
    elif hasattr(optim.lr_scheduler, scheduler_name):
        lr_scheduler = getattr(optim.lr_scheduler, scheduler_name)(
            optimizer, **{k:v for k, v in config.items() if k != "name"}
        )
    else:
        lr_scheduler = None
    
    if (logger is not None) and (lr_scheduler is not None):
        logger.info('Created [{}] from Config({})'.format(scheduler_name, config))
    return lr_scheduler