# from vqmap.models.mono_model import MONO
# from vqmap.models.mono_v2 import MONO_v2
from vqmap.models.seqvae import *
from vqmap.models.single_modal import *
from vqmap.models.gpt import GPT
from vqmap.comparisons.vae import get_vae
from vqmap.comparisons.latent import get_latentvae
from vqmap.comparisons.action2motion import get_action2motion
from vqmap.comparisons.denoising import DenoiseAutoencoder
from vqmap.comparisons.forecast.classifer import ForecastNet

__all__ = ['get_model']


def initialize_model(config):
    model_name = config.name
    model_cls = globals().get(model_name, None)
    assert model_cls, f"The specified model {model_name} was not found."
    
    model = model_cls(config)
    
    return model


def get_model(config):
    model_name = config.name
    
    if model_name == "posegpt":
        gpt = GPT(config["gpt"])
        return gpt
    
    nfeats = config.nfeats
    latent_dim = config.latent_dim
    lambdas = config.lambdas
    vae = config.get("vae", False)
    
    if model_name == 'mono':
        # return MONO(
        #     motion_enc=config.motion_encoder,
        #     neuro_enc=config.neuro_encoder,
        #     motion_dec=config.motion_decoder,
        #     lambdas=config.lambdas
        # )
        import os
        from vqmap.config.config import parse_config
        
        ckpt_path = config['checkpoint']
        config_path = os.path.join(os.path.dirname(ckpt_path), 'parameters.yaml')
        config_motion = parse_config(config_path)["model"]

        motion_enc, motion_dec = config_motion["motion_encoder"], config_motion["motion_decoder"]
        vq = config_motion["vq"]
        
        neuro_enc = config["encoder"]
        lambdas = config["lambdas"]

        model = MONO_v2(
            config_motion["nfeats"], config_motion["latent_dim"],
            nfeats, latent_dim,
            motion_enc, neuro_enc, motion_dec,
            lambdas=lambdas, vq=vq
        )
        model.motion_encdec.load_state_dict(
            torch.load(ckpt_path)["model"]
        )
        print("Loaded pretrained motion VQVAE")
        
        return model
        
    elif model_name == "single_neuro":
        return SingleModalityVAE(
            nfeats, latent_dim,
            enc_config=config.get("neuro_encoder", {}),
            dec_config=config.get("motion_decoder", {}),
            lambdas=lambdas,
            modality='neural',
            vae=vae,
            vq=config.get("vq", None)
        )
    elif model_name == "single_motion":
        return SingleModalityVAE(
            nfeats, latent_dim,
            enc_config=config.get("motion_encoder", {}),
            dec_config=config.get("motion_decoder", {}),
            lambdas=lambdas,
            modality='motion',
            vae=vae,
            vq=config.get("vq", None)
        )
    elif model_name == "multivqvae":
        return MultiVQVAE(
            nfeats, latent_dim, 
            strides_t=config.get("strides_t", (3, 2)),
            downs_t=config.get("downs_t", (2, 2)),
            lambdas=lambdas,
            vq=config.get("vq", None)
        )
    elif model_name == "hierarchical_vq":
        # return HierarchicalVQVAE(config, nfeats, latent_dim, lambdas)
        return HierarchicalVQVAEv2(config, nfeats)
    elif model_name == "vqvae2":
        return VQVAE2(config, nfeats)
    elif model_name == "style":
        return StyleVQVAE(config, nfeats, latent_dim, lambdas)
    elif model_name == "multi_branches":
        return MultiEncodersVAE(config, nfeats, latent_dim, lambdas, 
                                skeleton_biases=config.get("skeleton_biases", None))
    elif model_name == "multi_branches_v2":
        return MultiEncoderVAEv2(config, nfeats, latent_dim, lambdas, vae=config.get("vae", False)) 
    elif model_name == 'step_transformer':
        return TransformerStep(config)
    elif model_name.startswith('comparison'):
        architecture = model_name.split('_')[-1]
        if 'latent' in model_name:
            return get_latentvae(architecture, nfeats, latent_dim, lambdas)
        if 'action2motion' in model_name:
            return get_action2motion(nfeats, nfeats, 256, "cuda", lambdas)
        return get_vae(architecture, nfeats, latent_dim, lambdas)
    elif model_name == "denoising":
        return DenoiseAutoencoder(nfeats, latent_dim)
    elif model_name == "classifier":
        return ForecastNet(1, n_cls=15, latent_dim=latent_dim)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
