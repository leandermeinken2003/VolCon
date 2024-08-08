"""Define ContextAwareImageModelBase."""
#pylint: disable=too-many-instance-attributes, dangerous-default-value, too-many-arguments, too-few-public-methods

import torch
from torch import nn

from model.resnet_block import ResNetBlock
from model.context_image_transformer import ContextImageTransformer


class ContextAwareImageModelBase(nn.Module): #pylint: disable=too-many-instance-attributes
    """Define ContextAwareImageModelBase.
    
    The ContextAwareImageModelBase is a model backbone used to encode a micrograph, based on a
    positive and a negative context example.
    """

    def __init__( #pylint: disable=too-many-locals
            self,
            pre_encoder_channels: list[int] = [16, 16, 32],
            filter_sizes_pre_encoder: list[int] = [3, 3, 3],
            context_embedding_size: int = 128,
            context_embedding_mlp_depth: int = 2,
            context_aware_image_encoder_channels: list[int] = [32, 32, 32],
            filter_sizes_context_aware_image_encoder: list[int] = [3, 3, 5, 7],
            context_aware_image_encoder_block_depth: int = 2,
            received_context_embedding_size: int = 32,
            context_aware_image_encoder_pooling_size: int = 2,
            num_context_aware_blocks_per_pooling: int = 2,
            end_image_encoder_channels: list[int] = [64, 64, 128],
            filter_sizes_end_image_encoder: list[int] = [3, 5, 7],
            end_image_encoder_pooling_size: int = 2,
            num_end_encoder_blocks_per_pooling: int = 2,
            end_image_embedding_size: int = 256,
    ) -> None:
        """Define ContextAwareImageModelBase.
        
        :param pre_encoder_channels: A list of numbers that defines the number of encoder channels
        for each ResNet block in the pre-encoder.
        :param filter_sizes_pre_encoder: A list of numbers that defines the size of the filters
        used in the corresponding ResNet block of the pre-encoder.
        :param context_embedding_size: Define the size of the context embedding used in the context-
        image-transformer block.
        :param context_embedding_mlp_depth: The number of dense layers used in every context-encoder
        block.
        :param context_aware_image_encoder_channels: Define the number of output channels for each
        context-aware-image-encoder-block. The number of channels for each of the context-aware-
        image-encoder-channels is the context_number of total channels divided by the number of
        different channels.
        :param filter_sizes_context_aware_image_encoder: Defines the different filter sizes for each
        of the different context-aware-image-encoder-channels. Therefore the length of the list
        defining the filter sizes defines the number of context-aware-image-encoder-channels.
        :param context_aware_image_encoder_block_depth: Defines the depth of the context-aware-
        image-encoder-channels.
        :param received_context_embedding_size: Defines the size of the embedding used to create the
        dynamic filters of the context-aware-image-encoder-block.
        :param context_aware_image_encoder_pooling_size: The size of the pooling filter used between
        the image-context-transformers.
        :param num_context_aware_blocks_per_pooling: Number of blocks between each pooling layer.
        There is no pooling layer infront of the first context-image-transformer.
        :param end_image_encoder_channels: A list of numbers that defines the number of encoder
        channels for ResNet block in the end-encoder.
        :param filter_sizes_end_image_encoder: A list of numbers that defines the sizes of the
        filters used in each ResNet block in the end-encoder.
        :param end_image_encoder_pooling_size: The pooling size used in the end-encoder.
        :param num_end_encoder_blocks_per_pooling: The number of ResNet blocks between each pooling
        layer. There is no pooling layer infront of the first ResNet block of the end-encoder.
        :param end_image_embedding_size: The size of the embedding, which is outputed from the
        ContextAwareImageModelBase.
        """
        super().__init__()
        self._create_pre_encoder(
            pre_encoder_channels, filter_sizes_pre_encoder,
        )
        self.context_embedding_size_transformer = nn.Linear(
            pre_encoder_channels[-1], context_embedding_size,
        )
        self._create_context_image_transformer_blocks(
            context_embedding_size,
            context_embedding_mlp_depth,
            context_aware_image_encoder_channels,
            filter_sizes_context_aware_image_encoder,
            context_aware_image_encoder_block_depth,
            received_context_embedding_size,
            context_aware_image_encoder_pooling_size,
            num_context_aware_blocks_per_pooling,
        )
        self._create_end_image_encoder(
            end_image_encoder_channels,
            filter_sizes_end_image_encoder,
            end_image_encoder_pooling_size,
            num_end_encoder_blocks_per_pooling,
        )
        if self.end_encoder_stack:
            embedding_size_transformer_input_size = end_image_encoder_channels[-1]
        else:
            embedding_size_transformer_input_size = context_aware_image_encoder_channels[-1]
        self.image_embedding_size_transformer = nn.Linear(
            embedding_size_transformer_input_size, end_image_embedding_size,
        )
        self.image_embedding_size = end_image_embedding_size
        self.context_embedding_size = context_embedding_size

    def _create_pre_encoder(
            self, pre_encoder_channels: list[int], filter_sizes_pre_encoder: list[int],
    ) -> None:
        """Create a single pre-encoder for context images."""
        self.pre_encoder_stack = []
        input_channels = 1
        for output_channels, filter_size in zip(pre_encoder_channels, filter_sizes_pre_encoder):
            pre_encoder_block = ResNetBlock(input_channels, output_channels, filter_size)
            self.pre_encoder_stack.append(pre_encoder_block)
            input_channels = output_channels
        self.pre_encoder_stack = nn.Sequential(*self.pre_encoder_stack)

    def _create_context_image_transformer_blocks(
            self,
            context_embedding_size: int,
            context_embedding_mlp_depth: int,
            context_aware_image_encoder_channels: list[int],
            filter_sizes_context_aware_image_encoder: list[int],
            context_aware_image_encoder_block_depth: int,
            received_context_embedding_size: int,
            context_aware_image_encoder_pooling_size: int,
            num_context_aware_blocks_per_pooling: int,
    ) -> None:
        """Create ContextAwareInceptionBlocks which encode the images with context."""
        self.context_aware_encoder_stack = nn.ModuleList()
        input_image_encoder_channels = self.pre_encoder_stack[-1].output_channels
        for context_block_index, output_image_encoder_channels in enumerate(
            context_aware_image_encoder_channels,
        ):
            context_aware_inception_block = ContextImageTransformer(
                context_embedding_size,
                context_embedding_mlp_depth,
                input_image_encoder_channels,
                output_image_encoder_channels,
                filter_sizes_context_aware_image_encoder,
                context_aware_image_encoder_block_depth,
                received_context_embedding_size,
            )
            input_image_encoder_channels = context_aware_inception_block \
                .output_channels_image_encoder
            self.context_aware_encoder_stack.append(context_aware_inception_block)
            if not context_block_index + 1 % num_context_aware_blocks_per_pooling:
                self.context_aware_encoder_stack.append(context_aware_image_encoder_pooling_size)

    def _create_end_image_encoder(
            self,
            end_image_encoder_channels: list[int],
            filter_sizes_end_image_encoder: list[int],
            end_encoder_pooling_size: int,
            num_end_encoder_blocks_per_pooling: int,
    ) -> None:
        """Create end-image-encoder which encodes the model after the context has been encoded."""
        self.end_encoder_stack = []
        input_channels = self.context_aware_encoder_stack[-1].output_channels_image_encoder
        for end_encoder_index, (output_channels, filter_size) in enumerate(zip(
            end_image_encoder_channels, filter_sizes_end_image_encoder,
        )):
            end_encoder_block = ResNetBlock(input_channels, output_channels, filter_size)
            input_channels = output_channels
            self.end_encoder_stack.append(end_encoder_block)
            if not end_encoder_index + 1 % num_end_encoder_blocks_per_pooling:
                self.context_aware_encoder_stack.append(end_encoder_pooling_size)
        self.end_encoder_stack = nn.Sequential(*self.end_encoder_stack)

    def forward(
            self,
            image: torch.Tensor,
            context: torch.Tensor,
            contrastive_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the image using the provided context."""
        image, context, contrastive_context = self._pre_encode(
            image, context, contrastive_context,
        )
        image, context = self._context_aware_encoding(image, context, contrastive_context)
        image = self.end_encoder_stack(image)
        image_embedding = image.mean([2, 3])
        return self.image_embedding_size_transformer(image_embedding), context

    def _pre_encode(
            self,
            image: torch.Tensor,
            context: torch.Tensor,
            contrastive_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor]]:
        """Encode the context (+ image) before it is passed to the context-aware-encoder."""
        image = self.pre_encoder_stack(image)
        context = self.pre_encoder_stack(context)
        contrastive_context = self.pre_encoder_stack(contrastive_context)
        context = self.context_embedding_size_transformer(context.mean([-2, -1]))
        contrastive_context = self.context_embedding_size_transformer(
            contrastive_context.mean([-2, -1]),
        )
        return image, context, contrastive_context

    def _context_aware_encoding(
            self,
            image: torch.Tensor,
            context: torch.Tensor,
            contrastive_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode Image under the provided context."""
        for context_aware_encoding_block in self.context_aware_encoder_stack:
            if isinstance(context_aware_encoding_block, nn.AvgPool2d):
                image = context_aware_encoding_block(image)
            else:
                image, context, contrastive_context = context_aware_encoding_block(
                    image, context, contrastive_context,
                )
        return image, context
