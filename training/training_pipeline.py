"""Train the model given a set of hyperparameters."""
#pylint: disable=import-error

import os
import json
import random
from glob import glob
from collections import namedtuple

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from joblib.externals.loky import cpu_count

from model.volume_fraction_model import VolumeFractionModel
from model.context_model import ContextAwareImageModelBase
from training.generate_synthetic_two_phase_image import (
    generate_two_phase_image,
    ImageDimensions,
)
from utils.memory_saving import (
    remove_gpu_copies,
    clear_memory,
)
from utils.metrics import process_metrics
from utils.parallel_computing import (
    parallel_execution_of_function_with_same_params,
    parallel_execution_of_function_with_param_list,
)
from utils.reproducability import ensure_reproducability


TESTRUN_ID = '5.7M_Context_Linear1'
TESTRUN_PATH = f'testruns/{TESTRUN_ID}/'
MODEL_SAVE_PATH = TESTRUN_PATH + 'model.pth'
OPTIMIZER_SAVE_PATH = TESTRUN_PATH + 'optimizer.pth'
TRAIN_METRICS_PATH = TESTRUN_PATH + 'train_metrics.json'
TEST_METRICS_PATH = TESTRUN_PATH + 'test_metrics.json'
TESTDATA_DIRECTORY = 'testing/processed_testing_data/'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#data hyperparameters
DATA_HYPERPARAMETERS = {
    'shapes_type': ['gaussian_smoothed', 'stripe'],
    'combine_shape_types': 0.5,
    'alpha_augmentation': 0.3,
    'noise_augmentation': 0.5,
    'anomaly_patch_augmentation': 0.5,
    'stripe_augmentation': 0.1,
}
MAX_IMAGE_DIM = 384
MIN_IMAGE_DIM = 256

#model hyperparameters
BACKBONE_PARAMS = {
    'pre_encoder_channels': [32, 32, 32],
    'filter_sizes_pre_encoder': [3, 3, 3],
    'context_embedding_size': 128,
    'context_embedding_mlp_depth': 2,
    'context_aware_image_encoder_channels': [64, 64, 64, 128, 128, 128],
    'filter_sizes_context_aware_image_encoder': [3, 3, 5, 7],
    'context_aware_image_encoder_block_depth': 2,
    'received_context_embedding_size': 64,
    'context_aware_image_encoder_pooling_size': 2,
    'num_context_aware_blocks_per_pooling': 6,
    'end_image_encoder_channels': [256, 256, 256],
    'filter_sizes_end_image_encoder': [3, 3, 3],
    'end_image_encoder_pooling_size': 2,
    'end_image_embedding_size': 256,
}
USE_CONTEXT_LINEAR = True

#Training hyperparameters
LOSS_FUNCTION = nn.MSELoss()
EPOCHS = 5360
NUM_EPOCHS_EVAL = int(1e2)
STEPS_PER_EPOCH = 4
BATCH_SIZE = 2
LR = 1e-4


TrainingData = namedtuple('TrainingData', ['model_inputs', 'true_volume_fractions'])
TestCase = namedtuple('TestCase', ['model_inputs', 'true_volume_fractions'])


def main() -> None:
    """Run train and evaluation pipeline."""
    volume_fraction_model, optimizer = _init_training_setup()
    _save_hyperparameters()
    for epoch in tqdm(range(EPOCHS)):
        _epoch_training_step(volume_fraction_model, epoch, optimizer)
        if not epoch % NUM_EPOCHS_EVAL:
            _epoch_evaluation_step(volume_fraction_model, epoch)
        _save_optimizer_state_dict(optimizer)
        torch.save(volume_fraction_model, MODEL_SAVE_PATH)
        clear_memory()


def _init_training_setup() -> tuple[VolumeFractionModel, Adam]:
    ensure_reproducability()
    volume_fraction_model = _get_volume_fraction_model()
    optimizer = _get_optimizer(volume_fraction_model)
    return volume_fraction_model, optimizer


def _get_volume_fraction_model() -> VolumeFractionModel:
    if os.path.exists(MODEL_SAVE_PATH):
        volume_fraction_model = torch.load(MODEL_SAVE_PATH)
    else:
        volume_fraction_model = _create_volume_fraction_model()
        os.makedirs(TESTRUN_PATH)
    return volume_fraction_model.to(DEVICE)


def _create_volume_fraction_model() -> VolumeFractionModel:
    backbone = ContextAwareImageModelBase(**BACKBONE_PARAMS)
    return VolumeFractionModel(backbone, use_context_linear=USE_CONTEXT_LINEAR)


def _get_optimizer(volume_fraction_model: VolumeFractionModel) -> Adam:
    optimizer = Adam(volume_fraction_model.parameters(), lr=LR)
    if os.path.exists(OPTIMIZER_SAVE_PATH):
        optimizer = _load_optimizer_state_dict(optimizer)
    return optimizer


def _load_optimizer_state_dict(optimizer: Adam) -> dict:
    state_dict = torch.load(OPTIMIZER_SAVE_PATH)
    optimizer.load_state_dict(state_dict)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    return optimizer


def _save_hyperparameters() -> None:
    with open(TESTRUN_PATH + 'data_hyperparameters.json', 'w', encoding='utf-8') as file:
        json.dump(DATA_HYPERPARAMETERS, file)
    with open(TESTRUN_PATH + 'model_hyperparameters.json', 'w', encoding='utf-8') as file:
        json.dump(BACKBONE_PARAMS, file)


def _epoch_training_step(
        volume_fraction_model: VolumeFractionModel, epoch: int, optimizer: Adam,
) -> None:
    volume_fraction_model.train()
    optimizer.zero_grad()
    true_volume_fractions = []
    predicted_volume_fractions = []
    epoch_workers, step_workers = _split_workers_for_training_data_generation()
    training_data = parallel_execution_of_function_with_same_params(
        STEPS_PER_EPOCH, epoch_workers, _get_training_data, step_workers,
    )
    for training_batch in training_data:
        _one_step_in_epoch(
            volume_fraction_model,
            training_batch,
            true_volume_fractions,
            predicted_volume_fractions,
        )
    optimizer.step()
    true_volume_fractions = torch.concat(true_volume_fractions, dim=0)
    predicted_volume_fractions = torch.concat(predicted_volume_fractions, dim=0)
    process_metrics(
        true_volume_fractions, predicted_volume_fractions, epoch, TRAIN_METRICS_PATH, train=True,
    )
    remove_gpu_copies(true_volume_fractions, predicted_volume_fractions)


def _split_workers_for_training_data_generation() -> tuple[int, int]:
    cpu_cores = cpu_count()
    epoch_workers = cpu_cores // BATCH_SIZE
    return epoch_workers, BATCH_SIZE


def _get_training_data(step_workers: int) -> TrainingData:
    true_volume_fractions = []
    model_inputs = []
    image_dimensions = _get_image_dimensions()
    training_samples = parallel_execution_of_function_with_same_params(
        BATCH_SIZE, step_workers, _generate_one_training_sample, image_dimensions,
    )
    model_inputs = [training_sample[0] for training_sample in training_samples]
    true_volume_fractions = [training_sample[1] for training_sample in training_samples]
    model_inputs = _flatten_list(model_inputs)
    true_volume_fractions = _flatten_list(true_volume_fractions)
    model_inputs = _format_model_inputs(model_inputs)
    true_volume_fractions = torch.as_tensor(
        true_volume_fractions, dtype=torch.float32, device=DEVICE,
    )
    return TrainingData(model_inputs, true_volume_fractions)


def _get_image_dimensions() -> ImageDimensions:
    """Get the dimensions of the generated two phase image."""
    image_width = random.randint(MIN_IMAGE_DIM, MAX_IMAGE_DIM)
    image_height = random.randint(MIN_IMAGE_DIM, MAX_IMAGE_DIM)
    return ImageDimensions(image_height, image_width)


def _generate_one_training_sample(image_dimensions: ImageDimensions) -> None:
    microstructure_image, texture_vol_1, texture_vol_2 = generate_two_phase_image(
        image_dimensions,
        **DATA_HYPERPARAMETERS,
    )
    true_volume_fractions = [[texture_vol_1.volume_fraction], [texture_vol_2.volume_fraction]]
    model_inputs = [
        (microstructure_image, texture_vol_1, texture_vol_2), # input for first volume fraction
        (microstructure_image, texture_vol_2, texture_vol_1), # input for second volume fraction
    ]
    return model_inputs, true_volume_fractions


def _flatten_list(nested_list: list[list]) -> list:
    return [item for sublist in nested_list for item in sublist]


def _format_model_inputs(model_inputs: list[tuple]) -> dict[str, torch.Tensor]:
    microstructure_images = torch.concat(
        [model_input[0] / 255 for model_input in model_inputs], dim=0,
    ).to(DEVICE)
    context_1_images = _format_context_images(
        [model_input[1].texture_image for model_input in model_inputs],
    )
    context_2_images = _format_context_images(
        [model_input[2].texture_image for model_input in model_inputs],
    )
    # add the image channel channel to the tensors
    return {
        'microstructure_images': microstructure_images.unsqueeze(dim=1),
        'texture_images': context_1_images.unsqueeze(dim=1),
        'contrastive_texture_images': context_2_images.unsqueeze(dim=1),
    }


def _format_context_images(context_images: list[torch.Tensor]) -> torch.Tensor:
    min_width = min(context_image.shape[1] for context_image in context_images)
    min_height = min(context_image.shape[0] for context_image in context_images)
    new_context_images = []
    for context_image in context_images:
        image_width = context_image.shape[1]
        image_height = context_image.shape[0]
        start_x = np.random.randint(0, image_width - min_width) if image_width > min_width else 0
        start_y = np.random.randint(0, image_height - min_height) \
            if image_height > min_height else 0
        cropped_texture_image = context_image[start_y:start_y+min_height, start_x:start_x+min_width]
        normed_texture_image = cropped_texture_image / 255
        new_context_images.append(normed_texture_image.unsqueeze(dim=0))
    return torch.concat(new_context_images, dim=0).to(DEVICE)


def _one_step_in_epoch(
        volume_fraction_model: VolumeFractionModel,
        training_batch: TrainingData,
        true_volume_fractions_list: list,
        predicted_volume_fractions_list: list,
) -> None:
    predicted_volume_fractions = volume_fraction_model(**training_batch.model_inputs)
    loss = LOSS_FUNCTION(training_batch.true_volume_fractions, predicted_volume_fractions)
    loss /= STEPS_PER_EPOCH
    clear_memory()
    loss.backward()
    true_volume_fractions_list.append(training_batch.true_volume_fractions)
    predicted_volume_fractions_list.append(predicted_volume_fractions)
    remove_gpu_copies(loss)


def _epoch_evaluation_step(volume_fraction_model: VolumeFractionModel, epoch: int) -> None:
    volume_fraction_model.eval()
    testcase_directories = glob(TESTDATA_DIRECTORY + '*')
    predicted_volume_fractions_list = []
    testcases = parallel_execution_of_function_with_param_list(
        -1, _get_testdirectory_data, testcase_directories,
    )
    testcases_inputs, true_volume_fractions_list = _format_testcases(testcases)
    for testcase_inputs in testcases_inputs:
        _evaluate_testcase_directory(
            testcase_inputs, volume_fraction_model, predicted_volume_fractions_list,
        )
    predicted_volume_fractions = torch.concat(predicted_volume_fractions_list, dim=0)
    true_volume_fractions = torch.as_tensor(true_volume_fractions_list, device=DEVICE)
    process_metrics(
        true_volume_fractions, predicted_volume_fractions, epoch, TEST_METRICS_PATH, train=False,
    )


def _get_testdirectory_data(testcase_directory: str) -> tuple[torch.Tensor]:
    volume_fraction_df = pd.read_csv(testcase_directory + '/volume_fractions.csv')
    current_true_volume_fractions = volume_fraction_df['volume_fraction'].tolist()
    true_volume_fractions = [[vf] for vf in current_true_volume_fractions]
    microstructure_images = _get_microstructure_images(testcase_directory)
    context_inputs = _get_texture_images(testcase_directory)
    return TestCase((microstructure_images, *context_inputs), true_volume_fractions)


def _format_testcases(testcases: list[TestCase]) -> tuple[tuple[torch.Tensor], list[list[float]]]:
    testcases_inputs = [testcase.model_inputs for testcase in testcases]
    true_volume_fractions_nested_list = [testcase.true_volume_fractions for testcase in testcases]
    true_volume_fractions = _flatten_list(true_volume_fractions_nested_list)
    return testcases_inputs, true_volume_fractions


def _evaluate_testcase_directory(
        testcase_inputs: tuple[torch.Tensor],
        volume_fraction_model: VolumeFractionModel,
        predicted_volume_fractions_list: list,
) -> None:
    microstructure_images, context_images, contrastive_images = testcase_inputs
    for microstructure_image, context_image, contrastive_image in zip(
        microstructure_images, context_images, contrastive_images
    ):
        with torch.no_grad():
            microstructure_image = microstructure_image.to(DEVICE)
            context_image = context_image.to(DEVICE)
            contrastive_image = contrastive_image.to(DEVICE)
            predicted_volume_fractions = volume_fraction_model(
                microstructure_image, context_image, contrastive_image,
            )
            predicted_volume_fractions_list.append(predicted_volume_fractions)
        remove_gpu_copies(microstructure_image, context_image, contrastive_image)


def _get_microstructure_images(testcase_directory: str) -> torch.Tensor:
    microstructure_image = (
        torch.load(testcase_directory + '/microstructure.pt', weights_only=True) / 255
    ).to(DEVICE)
    return [microstructure_image, microstructure_image]


def _get_texture_images(testcase_directory: str) -> tuple[torch.Tensor]:
    texture_1_sample = torch.load(
        testcase_directory + '/material_1_sample.pt', weights_only=True,
    ) / 255
    texture_2_sample = torch.load(
        testcase_directory + '/material_2_sample.pt', weights_only=True,
    ) / 255
    texture_1_sample = texture_1_sample.to(DEVICE)
    texture_2_sample = texture_2_sample.to(DEVICE)
    context_images = [texture_1_sample, texture_2_sample]
    contrastive_images = [texture_2_sample, texture_1_sample]
    return context_images, contrastive_images


def _save_optimizer_state_dict(optimizer: Adam) -> None:
    state_dict = optimizer.state_dict()
    torch.save(state_dict, OPTIMIZER_SAVE_PATH)


if __name__ == '__main__':
    main()
