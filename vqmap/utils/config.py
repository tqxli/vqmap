from omegaconf import OmegaConf
from loguru import logger


def parse_config(base_config):
    config = {}
    for name, cfg_path in base_config.items():
        if name == 'expdir':
            config['expdir'] = cfg_path
            continue
        
        if name == 'dataset' and OmegaConf.is_list(cfg_path):
            logger.debug(f"Coembedding datasets {cfg_path}")
            config['dataset'] = [OmegaConf.load(path) for path in cfg_path]
            continue

        config.update(OmegaConf.load(cfg_path))
    return config