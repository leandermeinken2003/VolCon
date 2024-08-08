"""Define Model to predict volume percentages."""
#pylint: disable=not-callable

import torch
from torch import nn
from torch.nn import functional as F

from model.context_model import ContextAwareImageModelBase
from utils.parameters import (
    create_parameters,
    initialize_parameters,
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
        microstructure_image_embedding, context_embedding = self.backbone(
            microstructure_images, texture_images, contrastive_texture_images,
        )
        if self.use_context_linear:
            microstructure_image_embedding = self.hidden_layer(
                microstructure_image_embedding, context_embedding,
            )
            outputs = self.output_layer(microstructure_image_embedding, context_embedding)
        else:
            microstructure_image_embedding = self.backbone(microstructure_images)
            microstructure_image_embedding = self.hidden_layer(microstructure_image_embedding)
            outputs = self.output_layer(microstructure_image_embedding)
        return outputs


class ContextLinear(nn.Module):
    """Define ContextLinear."""
    def __init__(self, input_channels: int, output_channels: int, embedding_size: int) -> None:
        """Defome ContextLinear.
        
        :param input_channels: Defines the number of image channels of the input image.
        :param output_channels: Defines the number of image channels of the output image.
        :param embedding_size: The size of the context embedding received.
        """
        super().__init__()
        self.weight_generation_tensor = create_parameters((output_channels, 1))
        initialize_parameters(self.weight_generation_tensor)
        self.context_reception_tensor = create_parameters((input_channels, embedding_size))
        initialize_parameters(self.context_reception_tensor)
        self.context_bias_tensor = create_parameters((output_channels, embedding_size))
        initialize_parameters(self.context_bias_tensor)

    def forward(self, inputs: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """Calculate a context aware linear layer."""
        context_reception = torch.matmul(
            self.context_reception_tensor, context_embedding.transpose(0, 1),
        ).transpose(0, 1)
        bias = torch.matmul(
            self.context_bias_tensor, context_embedding.transpose(0, 1),
        ).transpose(0, 1)
        return self._apply_context_to_each_sample(inputs, context_reception, bias)

    def _apply_context_to_each_sample(
            self, inputs: torch.Tensor, context_reception: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        num_embedding = context_reception.shape[0]
        new_embedding = []
        for encoding_index in range(num_embedding):
            image_embedding = inputs[encoding_index].unsqueeze(0)
            image_bias = bias[encoding_index]
            image_context_reception = context_reception[encoding_index].reshape(1, -1)
            image_weights = torch.matmul(self.weight_generation_tensor, image_context_reception)
            new_embedding.append(F.linear(image_embedding, image_weights, image_bias))
        return torch.concat(new_embedding, dim=0)
