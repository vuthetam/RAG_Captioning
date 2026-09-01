from torch import nn


def trainable_parameters(*modules: nn.Module):
    """Yield all parameters that require gradients across the given modules."""
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.data_ptr() not in seen:
                seen.add(parameter.data_ptr())
                yield parameter
