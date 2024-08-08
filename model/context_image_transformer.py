"""Define the ContextImageTransformer and the submodules."""
#pylint: disable=too-many-instance-attributes, dangerous-default-value, too-many-arguments, too-few-public-methods, not-callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from utils.parameters import (
    create_parameters,
    initialize_parameters,
)


class ContextImageTransformer(nn.Module):
    """Define ContextImageTransformer.

    The purpose of the ContextImageTransformer is to exchange information between the Image
    (Mikrograph) and the context, with the goal of adapting the image encoding to the context.
    """
    def __init__(
            self,
            context_embedding_size: int,
            context_embedding_mlp_depth: int,
            input_image_encoder_channels: int,
            output_image_encoder_channels: int,
            filter_sizes_context_aware_image_encoder: list[int],
            context_aware_image_encoder_block_depth: int,
            received_context_embedding_size: int,
    ) -> None:
        """Define ConceptInceptionBlock."""
        super().__init__()
        self._create_attention_layer(input_image_encoder_channels, context_embedding_size)
        self.context_encoder = ContextEncoderBlock(
            context_embedding_size, context_embedding_mlp_depth,
        )
        self._create_context_aware_image_encoder_block(
            input_image_encoder_channels,
            output_image_encoder_channels,
            received_context_embedding_size,
            filter_sizes_context_aware_image_encoder,
            context_aware_image_encoder_block_depth,
            context_embedding_size,
        )
        self.output_channels_image_encoder = self.context_aware_image_encoder.output_channels

    def _create_attention_layer(
            self, input_image_encoder_channels: int, context_embedding_size: int,
    ) -> None:
        self.image_embedding_size_to_context_embedding_size = nn.Linear(
            input_image_encoder_channels, context_embedding_size,
        )
        self.attention_layer = ImageContextAttentionBlock(context_embedding_size)

    def _create_context_aware_image_encoder_block(
            self,
            input_image_encoder_channels: int,
            output_image_encoder_channels: int,
            received_context_embedding_size: int,
            filter_sizes_context_aware_image_encoder: list[int],
            context_aware_image_encoder_block_depth: int,
            context_embedding_size: int,
    ) -> None:
        self.context_embedding_size_to_image_embedding_size = nn.Linear(
            context_embedding_size, received_context_embedding_size,
        )
        self.context_aware_image_encoder = ContextAwareImageEncoderBlock(
            input_image_encoder_channels,
            output_image_encoder_channels,
            filter_sizes_context_aware_image_encoder,
            context_aware_image_encoder_block_depth,
            received_context_embedding_size,
        )

    def forward(
            self,
            image: torch.Tensor,
            context_embedding: torch.Tensor,
            contrastive_context_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one block of context_encoder + context_aware_image_encoder."""
        context_embedding, contrastive_context_embedding = self._apply_attention_layer(
            image, context_embedding, contrastive_context_embedding,
        )
        image = self._apply_context_aware_image_encoder(image, context_embedding)
        return (
            image,
            context_embedding,
            contrastive_context_embedding,
        )

    def _apply_attention_layer(
            self,
            image: torch.Tensor,
            context_embedding: torch.Tensor,
            contrastive_context_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = image.mean([-2, -1])
        image_embedding = self.image_embedding_size_to_context_embedding_size(image_embedding)
        image_embedding = F.sigmoid(image_embedding)
        context_embedding, contrastive_context_embedding = self.attention_layer(
            image_embedding, context_embedding, contrastive_context_embedding,
        )
        return context_embedding, contrastive_context_embedding

    def _apply_context_aware_image_encoder(
            self, image: torch.Tensor, context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        squashed_context_embedding = self.context_embedding_size_to_image_embedding_size(
            context_embedding,
        )
        num_images = image.shape[0]
        new_image = []
        for image_index in range(num_images):
            selected_image = image[image_index]
            selected_context_embedding = squashed_context_embedding[image_index]
            selected_image = self.context_aware_image_encoder(
                selected_image, selected_context_embedding,
            )
            new_image.append(selected_image)
        return torch.concat(new_image, dim=0)


class ImageContextAttentionBlock(nn.Module):
    """Define ContextAttentionBlock.
    
    Define an attention block, which encodes the relationship between the context embedding and the
    image-embedding.
    """
    def __init__(self, embedding_size: int) -> None:
        """Define ContextAttentionBlock.
        
        :param embedding_size: The size of the used embeddings.
        """
        super().__init__()
        self._create_image_to_context_attention_operation(embedding_size)
        self._create_context_to_context_attention_operation(embedding_size)

    def _create_image_to_context_attention_operation(self, embedding_size: int) -> None:
        self.image_context_query_matrix = nn.Linear(embedding_size, embedding_size)
        self.image_context_key_matrix = nn.Linear(embedding_size, embedding_size)
        self.image_context_value_matrix = nn.Linear(embedding_size, embedding_size)

    def _create_context_to_context_attention_operation(self, embedding_size: int) -> None:
        self.context_context_query_matrix = nn.Linear(embedding_size, embedding_size)
        self.context_context_key_matrix = nn.Linear(embedding_size, embedding_size)
        self.context_context_value_matrix = nn.Linear(embedding_size, embedding_size)

    def forward(
            self,
            image_embedding: torch.Tensor,
            context_embedding: torch.Tensor,
            contrastive_context_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the ImageContextAttention block."""
        new_context_embedding = self._apply_attention_to_embedding_vector(
            context_embedding, contrastive_context_embedding, image_embedding,
        )
        new_contrastive_context_embedding = self._apply_attention_to_embedding_vector(
            contrastive_context_embedding, context_embedding, image_embedding,
        )
        return new_context_embedding, new_contrastive_context_embedding

    def _apply_attention_to_embedding_vector(
            self,
            first_context_embedding: torch.Tensor,
            second_context_embedding: torch.Tensor,
            image_embedding: torch.Tensor,
    ) -> torch.Tensor:
        image_context_weights = self._get_image_context_attention_weight(
            first_context_embedding, image_embedding,
        )
        context_context_weights = self._get_context_context_attention_weight(
            first_context_embedding, second_context_embedding,
        )
        image_context_weights, context_context_weights = self._adjust_weights(
            image_context_weights, context_context_weights,
        )
        image_context_value_vector = self.image_context_value_matrix(image_embedding)
        context_context_value_vector = self.context_context_value_matrix(second_context_embedding)
        return first_context_embedding + \
            image_context_weights * image_context_value_vector + \
            context_context_weights * context_context_value_vector

    def _get_image_context_attention_weight(
            self, context_embedding: torch.Tensor, image_embedding: torch.Tensor,
    ) -> torch.Tensor:
        query_vector = self.image_context_query_matrix(image_embedding)
        key_vectors = self.image_context_key_matrix(context_embedding)
        cosine_similarity = self._calculate_batch_cosine_similarity(query_vector, key_vectors)
        return cosine_similarity.view(cosine_similarity.shape[0], 1)

    def _get_context_context_attention_weight(
            self, first_context_embedding: torch.Tensor, second_context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        query_vector = self.context_context_query_matrix(second_context_embedding)
        key_vectors = self.context_context_key_matrix(first_context_embedding)
        cosine_similarity = self._calculate_batch_cosine_similarity(query_vector, key_vectors)
        return cosine_similarity.view(cosine_similarity.shape[0], 1)

    def _calculate_batch_cosine_similarity(
            self, first_embedding: torch.Tensor, second_embedding: torch.Tensor,
    ) -> torch.Tensor:
        first_embedding_norm = torch.norm(first_embedding, dim=-1)
        second_embedding_norm = torch.norm(second_embedding, dim=-1)
        embedding_dotproduct = torch.sum(first_embedding * second_embedding, dim=-1)
        return embedding_dotproduct / (first_embedding_norm * second_embedding_norm)

    def _adjust_weights(
            self, image_context_weight: float, context_context_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_weight = image_context_weight + context_context_weight
        total_weight_mask = total_weight > 1
        image_context_weight[total_weight_mask] = \
            image_context_weight[total_weight_mask] / total_weight[total_weight_mask]
        context_context_weight[total_weight_mask] = \
            context_context_weight[total_weight_mask] / total_weight[total_weight_mask]
        return image_context_weight, context_context_weight


class ContextEncoderBlock(nn.Module):
    """Define ContextEncoderBlock.

    The ContextEncoderBlock encodes the old in combination with the new information, after the
    ImageContextAttentionBlock.
    """
    def __init__(
            self,
            context_embedding_size: int,
            context_embedding_mlp_depth: int,
    ) -> None:
        """Define ContextEncoderBlock.
        :param context_embedding_size: Defines the size of the embedding used.
        :param context_embedding_mlp_depth: Defines the number of layers in the ContextEncoderBlock.
        """
        super().__init__()
        context_encoder_list = []
        for _ in range(context_embedding_mlp_depth-1):
            context_encoder_list.extend([
                nn.Linear(context_embedding_size, context_embedding_size),
                nn.ReLU(),
            ])
        self.context_encoder = nn.Sequential(*context_encoder_list)

    def forward(
            self,
            context_embedding: torch.Tensor,
            contrastive_context_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode received context and contrastive context embeddings."""
        context_embedding = self.context_encoder(context_embedding)
        contrastive_context_embedding = self.context_encoder(contrastive_context_embedding)
        return context_embedding, contrastive_context_embedding


class ContextAwareImageEncoderBlock(nn.Module):
    """Define ContextAwareImageEncoderBlock.
    
    The ContextAwareImageEncoderBlock encodes the received image based on the received context, in
    order to enable a prediction adapted to the context.
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filter_sizes: list[int],
            context_aware_image_encoder_block_depth: int,
            context_embedding_size: int,
    ) -> None:
        """Define ContextAwareImageEncoderInceptionBlock.
        
        :param input_channels: The number of image channels of the input image.
        :param output_channels: The maximum total number of output image channels. The real number
        of output channels per ImageEncoderChannel is output_channels // len(filter_sizes).
        :param filter_sizes: A list of numbers that defines the size of the filters used in the
        ImageEncoderChannels. For each entry in the list one ImageEncoderChannel is created.
        :param context_aware_image_encoder_block_depth: Defines the depth of all the
        ImageEncoderChannels.
        :param context_embedding_size: Defines the size of the embedding received by the
        ContextAwareConv2d layers in the ImageEncoder channels.
        """
        super().__init__()
        output_channels_per_filter, output_channels = self._get_output_channels_per_filter(
            output_channels, filter_sizes,
        )
        self.filter_size_channels = nn.ModuleList()
        for filter_size in filter_sizes:
            filter_size_channel = ContextAwareConvolutionChannel(
                input_channels,
                output_channels_per_filter,
                filter_size,
                context_embedding_size,
                context_aware_image_encoder_block_depth,
            )
            self.filter_size_channels.append(filter_size_channel)
        if not output_channels == input_channels:
            self.upsample = nn.Conv2d(input_channels, output_channels, 1)
        else:
            self.upsample = None
        self.output_channels = output_channels

    def _get_output_channels_per_filter(self, output_channels: int, filter_sizes: list[int]) -> int:
        """Get the number of output channels per filter sizes used and the actual output size."""
        num_filters = len(filter_sizes)
        output_channels_per_filter = output_channels // num_filters
        return output_channels_per_filter, num_filters * output_channels_per_filter

    def forward(
            self, image_x: torch.Tensor, context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the ContextAwareImageEncoderBlock."""
        residual = image_x
        if self.upsample:
            residual = self.upsample(residual)
        image_x = [
            convolution_channel(image_x, context_embedding)
            for convolution_channel in self.filter_size_channels
        ]
        image_x = torch.concat(image_x, dim=0)
        image_x += residual
        image_x = F.relu(image_x)
        image_x = image_x.unsqueeze(0)
        return image_x


class ContextAwareConvolutionChannel(nn.Module):
    """Define ContextAwareConvolutionChannel."""
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filter_size: int,
            context_embedding_size: int,
            channel_depth: int,
    ) -> None:
        super().__init__()
        self.convolution_channel = nn.ModuleList()
        for channel_index in range(channel_depth):
            if channel_index:
                current_input = output_channels
            else:
                current_input = input_channels
            convolution_operation = ContextAwareConv2d(
                current_input,
                output_channels,
                filter_size,
                context_embedding_size,
                padding="same",
            )
            self.convolution_channel.append(convolution_operation)

    def forward(self, image_x: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """Apply a single context aware convolution channel."""
        num_operations = len(self.convolution_channel)
        for operation_idx, convolution_operation in enumerate(self.convolution_channel):
            image_x = convolution_operation(image_x, context_embedding)
            if not num_operations - 1 == operation_idx:
                image_x = F.relu(image_x)
        return image_x


class ContextAwareConv2d(_ConvNd):
    """Define ContextAwareConv2d.
    
    The ContextAwareConv2d module is a Conv2d Module, that dynamically defines the weights,
    of the filter dependant the context, which is passed to it.
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filter_size: int,
            embedding_size: int,
            padding: str | tuple[int] | None = None,
    ) -> None:
        """Define ContextAwareConv2d.
        
        :param input_channels: The number of input channels passed to the conv2d operation.
        :param output_channels: The number of output channels created by this conv2d operation.
        :param filter_size: The size of the filter used int he model.
        :param embedding_size: The size of the context-embedding passed to the conv2d layer.
        :param padding: The padding type used.
        """
        super().__init__(
            input_channels,
            output_channels,
            _pair(filter_size),
            stride=_pair(1),
            padding=padding,
            dilation=_pair(1),
            transposed=False,
            output_padding=_pair(0),
            groups=1,
            bias=False,
            padding_mode="zeros",
            device=None,
        )
        self.filter_generation_tensor = create_parameters(
            (output_channels, input_channels, filter_size, filter_size),
        )
        initialize_parameters(self.filter_generation_tensor)
        self.context_reception_tensor = create_parameters(
            (filter_size, filter_size, embedding_size),
        )
        initialize_parameters(self.context_reception_tensor)
        self.context_bias_tensor = create_parameters((output_channels, embedding_size),)
        initialize_parameters(self.context_bias_tensor)
        self.weight = None

    def forward(self, image: torch.Tensor, context_embedding: torch.Tensor) -> torch.Tensor:
        """Compute conv2d operation with context specific filter."""
        context_reception = torch.matmul(self.context_reception_tensor, context_embedding)
        context_weight = torch.matmul(self.filter_generation_tensor, context_reception)
        context_bias = torch.matmul(self.context_bias_tensor, context_embedding)
        return F.conv2d(
            image,
            context_weight,
            context_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
