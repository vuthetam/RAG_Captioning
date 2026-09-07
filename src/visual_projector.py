from torch import Tensor, nn


class VisualProjector(nn.Module):
    """Projects visual backbone features into the decoder embedding space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)
