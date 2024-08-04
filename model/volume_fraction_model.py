"""Define Model to predict volume percentages."""

import torch
from torch import nn

from volcon.models.context_model import (
    ContextAwareImageModelBase,
    ContextLinear,
)


class VolumeFractionModel(nn.Module):
    """Define VolumeFractionModel.
    
    This Model predict the volume fractions of a mikrostruktur defined by a context image. In
    addition to the context image, a contrastive context image can be added to identify, which
    phase not to predict the volume fraction for.
    """
    def __init__(
            self, backbone: ContextAwareImageModelBase, use_context_linear: bool = True,
    ) -> None:
        """Define VolumeFractionModel.
        
        :param backbone: The model backbone used to encode the context and mikrograph.
        :param use_context_linear: Decide if the ContextLinear layer should be used or a normal
        Linear layer.
        """
        super().__init__()
        self.backbone = backbone
        self.use_context_linear = use_context_linear
        mikrograph_embedding_size = backbone.image_embedding_size
        if self.use_context_linear:
            context_embedding_size = backbone.context_embedding_size
            self.hidden_layer = ContextLinear(
                mikrograph_embedding_size, mikrograph_embedding_size, context_embedding_size,
            )
            self.output_layer = ContextLinear(mikrograph_embedding_size, 1, context_embedding_size)
        else:
            self.hidden_layer = nn.Linear(mikrograph_embedding_size, mikrograph_embedding_size)
            self.output_layer = nn.Linear(mikrograph_embedding_size, 1)

    def forward(
            self,
            microstructure_images: torch.Tensor,
            texture_images: torch.Tensor,
            contrastive_texture_images: torch.Tensor,
    ) -> torch.Tensor:
        """Predict VolumeFraction for mikrograph and the corresponding context."""
        microstructure_image_encodings, context_embedding = self.backbone(
            microstructure_images, texture_images, contrastive_texture_images,
        )
        if self.use_context_linear:
            microstructure_image_encodings = self.hidden_layer(microstructure_image_encodings)
            outputs = self.output_layer(microstructure_image_encodings, context_embedding)
        else:
            microstructure_image_encodings = self.backbone(microstructure_images)
            microstructure_image_encodings = self.hidden_layer(microstructure_image_encodings)
            outputs = self.output_layer(microstructure_image_encodings)
        return outputs
