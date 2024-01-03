import os
from torch.utils.tensorboard import SummaryWriter
import fire
from omegaconf import OmegaConf
from loguru import logger

from vqmap.trainer.trainer import TrainerEngine
from vqmap.utils.run import set_random_seed
from vqmap.datasets import initialize_dataset

def main(config_path):
    config = OmegaConf.load(config_path)
    
    # create exp directory
    expdir = config.expdir
    if not os.path.exists(expdir):
        os.makedirs(expdir)
        os.makedirs(os.path.join(expdir, 'tb'))

    logger.add(
        os.path.join(expdir, f"stats_train.log"),
        format="{time:YYYY-MM-DD HH:mm} | {level} | {message}"
    )

    # initialize tensorboard logger
    tb_logger = SummaryWriter(os.path.join(expdir, 'tb'))
    
    # set random seed
    if config.get("seed", None) is not None:
        seed = config.seed
        set_random_seed(seed)
        logger.info(f"Set random seed to: {seed}\n")
    
    # initialize dataset & dataloaders
    dataloaders = initialize_dataset(config.dataset)

    # initialize trainer & model
    engine = TrainerEngine()
    engine.set_tb(tb_logger)
    engine.create(config)
    OmegaConf.save(config, os.path.join(config.expdir, 'parameters.yaml'))

    # start train
    engine.train(tr_loader=dataloaders.pop('train'),
                 n_epochs=config.train.epochs,
                 val_loaders=dataloaders,
                 val_epochs=config.train.get('val_epochs', 1),
                 model_save_to=os.path.join(config.expdir, config.train.model_save_path),
                 best_model_save_to=os.path.join(config.expdir, config.train.best_model_save_path)
    )
    
    return

if __name__ == '__main__':
    fire.Fire(main)