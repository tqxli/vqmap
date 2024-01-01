import pandas as pd

def flatten_dict(dict_, sep):
    return pd.json_normalize(dict_, sep=sep).to_dict(orient='records')[0]

def torch_safe_load(module, state_dict, strict=True):
    module.load_state_dict({
        k.replace('module.', ''): v for k, v in state_dict.items()
    }, strict=strict)