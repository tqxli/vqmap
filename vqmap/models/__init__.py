from vqmap.models.seqvae import *
from vqmap.models.gpt import GPT
from vqmap.comparisons.vae import get_vae
from vqmap.comparisons.latent import get_latentvae
from vqmap.comparisons.action2motion import get_action2motion
from vqmap.comparisons.denoising import DenoiseAutoencoder


def initialize_model(config):
    model_name = config.name
    
    # comparsion models: GRU, Transformer, ...
    # keep these here for now
    if model_name.startswith('comparison'):
        architecture = model_name.split('_')[-1]
        nfeats, latent_dim, lambdas = config.nfeats, config.latent_dim, config.lambdas
        if 'latent' in model_name:
            return get_latentvae(architecture, nfeats, latent_dim, lambdas)
        if 'action2motion' in model_name:
            return get_action2motion(nfeats, nfeats, 256, "cuda", lambdas)
        return get_vae(architecture, nfeats, latent_dim, lambdas)
    elif model_name == "denoising":
        return DenoiseAutoencoder(nfeats, latent_dim)
    
    # VQ based VAEs, Motion GPTs
    model_cls = globals().get(model_name, None)
    assert model_cls, f"The specified model {model_name} was not found."
    
    model = model_cls(config)
    
    return model
