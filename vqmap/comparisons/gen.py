import os
import numpy as np
import fire
from vqmap.comparisons.vae import get_vae
from vqmap.utils.run import set_random_seed
from vqmap.config.config import parse_config
from vqmap.logging.logger import PythonLogger
from vqmap.trainer.trainer import TrainerEngine
from vqmap.datasets.loaders import prepare_dataloaders, collate_fn, collate_fn_mocap
from vqmap.utils.visualize import visualize, visualize2d

def main(
    checkpoint, 
    seed=1024,
    n_samples=4, 
    **kwargs
):
    # load parameters from training directory for consistency
    expdir = os.path.dirname(checkpoint)
    config_path = os.path.join(expdir, 'parameters.yaml')
    config = parse_config(config_path)
    
    logger = PythonLogger(name=os.path.join(expdir, 'eval.txt'))
    
    # create model
    engine = TrainerEngine(device='cuda')
    engine.set_logger(logger)
    engine.create(config)
    engine.model_to_device()
    engine.load_state_dict(checkpoint, load_keys=['model'])
    engine.model.eval()
    
    duration = config.dataloader.seqlen
    print(f"Generate: sequence length {duration}")
    
    set_random_seed(seed)
    gen_seqs = np.zeros((n_samples, duration, 23, 3))
    for i in range(n_samples):
        gen = engine.model.generate(duration)
        gen_seqs[i] = gen.detach().cpu().numpy()
    
    savepath = os.path.join(expdir, f'visualize_gen_{seed}.mp4')

    anim = visualize(
        [gen_seqs],
        duration,
        savepath,
        ["Gen"]*n_samples
    )

if __name__ == "__main__":
    fire.Fire(main)