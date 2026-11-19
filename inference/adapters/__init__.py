"""Model-specific prediction functions with a common normalized-box result."""


def get_predictor(name):
    # Lazy imports keep evaluation and the dashboard independent of torch.
    if name == "grounding-dino":
        from .grounding_dino import predict
    elif name == "owlv2":
        from .owlv2 import predict
    elif name == "qwen3-vl":
        from .qwen3_vl import predict
    else:
        raise ValueError(f"Unknown model: {name}")
    return predict
