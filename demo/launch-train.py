from itertools import product
import os
from typing import Dict, List


def execute_command(command: List[str]):
    print(" ".join(command))
    os.system(" ".join(command))


def hparam_sweep(param_dict: Dict[str, List]):
    param_list = [
        [f"{k}={v}".replace(" ", "") for v in vs] for k, vs in param_dict.items()
    ]
    param_list = product(*param_list)
    param_list = [list(p) for p in param_list]
    return param_list


if __name__ == "__main__":
    # CHANGE expdir if you want to save to a different directory
    expdir = "models_local"

    # we follow the hydra convention for specifying config in the command line
    # check `configs/train.yaml` for more details
    base_command = ["python ../csbev/core/train.py"]
    base_command += [f"expdir={expdir}"]
    base_command += ["automatic_naming=True"]
    base_command += ["model/bottleneck=quantizer_mg_res"]
    
    # hyperparameter sweep can be enabled
    # if multiple parameters are specified for the same argument
    hparams = {        
        "train.augmentation.lr_flip_prob": [1.0],
        "model.lambdas.assignment": [0.01],
        "model.loss_cfg.assignment_delay": [0],
        "model.encoder.channel_encoder.n_ds": [4],
        "model.decoder.decoder_shared.n_ds": [4],
        "model.encoder.latent_dim": [128],
        "model.decoder.decoder_shared.in_channels": [128],
        "model.bottleneck.code_dim": [[64, 64]],
    }
    hparam_list = hparam_sweep(hparams)
    for hparam in hparam_list:
        execute_command(base_command + hparam)