from . import model_v0

MODELS = {
    "ModelV0": model_v0.ModelV0,
}


def get_model(cfg, obs_space, action_space, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
