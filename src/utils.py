import torch
from torch import nn


def trainable_parameters(*modules: nn.Module):
    """Yield all parameters that require gradients across the given modules."""
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.data_ptr() not in seen:
                seen.add(parameter.data_ptr())
                yield parameter


def extract_clip_features(output: object) -> torch.Tensor:
    """Extract projected CLIP embeddings from old and new Transformers outputs."""
    if isinstance(output, torch.Tensor):
        features = output
    else:
        features = None
        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attribute, None)
            if isinstance(value, torch.Tensor):
                features = value
                break

        # Tuple output can contain last_hidden_state before the pooled embedding.
        if features is None and isinstance(output, (tuple, list)):
            features = next(
                (value for value in output if isinstance(value, torch.Tensor) and value.ndim == 2),
                None,
            )

        if features is None:
            raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    if features.ndim != 2:
        raise ValueError(
            f"CLIP features phải có 2 chiều (batch, dim), nhận được {tuple(features.shape)}"
        )
    return features
