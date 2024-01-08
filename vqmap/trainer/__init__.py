from vqmap.trainer.trainer import TrainerEngine
from vqmap.trainer.trainer_gpt import TrainerEngineGPT


def initialize_trainer(config):
    if config.model.name == 'GPT':
        engine_cls = TrainerEngineGPT
    else:
        engine_cls = TrainerEngine
    engine = engine_cls()
    
    return engine