from omegaconf import OmegaConf
from vqmap.trainer.trainer import TrainerEngine
from vqmap.trainer.trainer_gpt import TrainerEngineGPT
from vqmap.trainer.trainer_multidatasets import TrainerEngineCoembed

def initialize_trainer(config):
    if config.model.name == 'GPT':
        engine_cls = TrainerEngineGPT
    elif OmegaConf.is_list(config.dataset):
        engine_cls = TrainerEngineCoembed
    else:
        engine_cls = TrainerEngine
    engine = engine_cls()
    
    return engine