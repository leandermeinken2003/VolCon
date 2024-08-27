"""Generate an image with two phases, that have different textures."""
#pylint: disable=no-member, import-error, too-many-arguments, too-many-locals, dangerous-default-value

import random
from collections import namedtuple

import cv2
import torch
import numpy as np
from numpy.random import default_rng
import skimage

from training.synthetic_data_hyperparameters.shapes import (
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    SMOOTHING_PROBABILITY,
    MAX_SHAPE_STRIPES,
    MIN_SHAPE_STRIPES,
    MAX_SHAPE_STRIPE_THICKNESS,
    MIN_SHAPE_STRIPE_THICKNESS,
)
from training.synthetic_data_hyperparameters.augmentation import (
    MAX_NOISE_STD,
    ANOMALY_PATCH_SHAPES,
    MAX_ANOMALY_PATCHES,
    MIN_ANOMALY_PATCHES,
    MAX_ANOMALY_RECT_DIM,
    MAX_ANOMALY_CIRCLE_RADIUS,
    MAX_RAND_ALPHA_PATCHES,
    MAX_ALPHA_PATCH_DIM,
    MAX_AUGMENT_STRIPES,
    MIN_AUGMENT_STRIPES,
    MAX_AUGMENT_STRIPE_THICKNESS,
    MIN_AUGMENT_STRIPE_THICKNESS,
)


DTYPE = torch.float32
MAX_ITER = 20
MAX_TEXTURE_SAMPLE_DIM = 25
MAX_TEXTURE_SAMPLE_SIZE = 400


ImageDimensions = namedtuple("ImageDimensions", ["height", "width"])
TextureVolFrac = namedtuple("TextureMask", ["texture_image", "volume_fraction"])


def generate_two_phase_image(
        image_dimensions: ImageDimensions,
        shapes_type: str | list[str] = ["gaussian_smoothed", "stripe"],
        combine_shape_types: bool | float = 0.5,
        min_phase_distance: int = MAX_NOISE_STD,
        alpha_augmentation: bool | float = 0.5,
        noise_augmentation: bool | float = 0.5,
        anomaly_patch_augmentation: bool | float = 0.5,
        stripe_augmentation: bool | float = 0.5,
    ) -> tuple[torch.Tensor, TextureVolFrac, TextureVolFrac]:
    """Generate a two phase image and apply the selected augmentations.
    
    :param image_dimensions: Set the dimensions of the generated image.
    :param shapes_type: Decide the kind of shapes that can be used in the two phase image. The type
    of shapes that can be used are gaussian_smoothed and/or stripe.
    :param combine_shape_types: If more than one shape is provided, decide, if to combine the shapes
    in one iamge.
    :param min_phase_distance: The minimum color distance between the different phases.
    :param alpha_augmentation: Decide wether to use alpha augmentation.
    :param noise_augmentation: Decide wether to use noise augmentation.
    :param anomaly_patch_augmentation: Decide wether to use anomaly patch augmentation.
    :param stripe_augmentation: Decide wether to use stripe augmentation.

    For all the augmenation options and the combine_shape_types parameter also a floating point
    value can be provided, which set the probability of use.
    """
    if isinstance(shapes_type, str):
        shapes_type = [shapes_type]
    if not _bool_or_random_check(combine_shape_types):
        selected_shape_type = random.choice(shapes_type)
        shapes_type = [selected_shape_type]
    two_phase_image, phase_1_mask, phase_2_mask = _create_two_phase_base_image(
        min_phase_distance, shapes_type, image_dimensions,
    )
    two_phase_image = _augment_two_phase_image(
        two_phase_image,
        noise_augmentation,
        alpha_augmentation,
        anomaly_patch_augmentation,
        stripe_augmentation,
        phase_1_mask,
        phase_2_mask,
        image_dimensions,
    )
    texture_vol_frac_1, texture_vol_frac_2 = _get_texture_volume_fractions(
        two_phase_image, phase_1_mask, phase_2_mask, image_dimensions,
    )
    two_phase_image = torch.as_tensor(
        two_phase_image.reshape((1, image_dimensions.height, image_dimensions.width)),
        dtype=DTYPE,
    )
    return two_phase_image, texture_vol_frac_1, texture_vol_frac_2


def _create_two_phase_base_image(
        min_phase_distance: int, shapes_type: list, image_dimensions: ImageDimensions,
) -> np.ndarray:
    phase_1_color, phase_2_color = _get_phase_colors(min_phase_distance)
    phase_1_mask = _get_phase_1_mask(shapes_type, image_dimensions)
    phase_2_mask = _get_phase_2_mask(phase_1_mask)
    image_part_phase_1 = phase_1_color * phase_1_mask
    image_part_phase_2 = phase_2_color * phase_2_mask
    two_phase_image = image_part_phase_1 + image_part_phase_2
    return two_phase_image, phase_1_mask, phase_2_mask


def _get_phase_colors(min_phase_distance: int) -> tuple[int, int]:
    assert min_phase_distance >= 0, "The min_phase_distance has to be a positive value."
    phase_1_color = random.randint(min_phase_distance + 1, 255)
    phase_2_color = random.randint(0, phase_1_color - min_phase_distance)
    phase_colors = [phase_1_color, phase_2_color]
    random.shuffle(phase_colors)
    return phase_colors


def _get_phase_1_mask(shapes_types: list[str], image_dimensions: ImageDimensions) -> np.ndarray:
    volume_fraction_phase_1 = np.random.random() * 0.90 + 0.05
    phase_1_mask = np.zeros((image_dimensions.height, image_dimensions.width), dtype=np.int32)
    total_mask_pixels = image_dimensions.height * image_dimensions.width
    for _ in range(MAX_ITER):
        shape_type = random.choice(shapes_types)
        if shape_type == "gaussian_smoothed":
            phase_1_mask += _create_gaussian_smoothed_shapes(image_dimensions)
        elif shape_type == "stripe":
            phase_1_mask += _create_shape_stripes(image_dimensions)
        phase_1_mask = np.where(phase_1_mask > 0, 1, 0)
        if _is_volume_fraction_exceeded(phase_1_mask, total_mask_pixels, volume_fraction_phase_1):
            break
    return phase_1_mask


def _create_gaussian_smoothed_shapes(image_dimensions: ImageDimensions) -> np.ndarray:
    random_pixel_image = _generate_random_pixel_image(image_dimensions)
    smoothed_image = cv2.GaussianBlur(
        random_pixel_image, ksize=(0,0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT,
    )
    mask = _apply_threshold_to_smoothed_random_image(smoothed_image)
    return _smooth_shapes(mask)


def _generate_random_pixel_image(image_dimensions: ImageDimensions) -> np.ndarray:
    rng = default_rng()
    return rng.integers(
        low=0,
        high=255,
        size=(image_dimensions.height, image_dimensions.width),
        dtype=np.uint8,
        endpoint=True,
    )


def _apply_threshold_to_smoothed_random_image(smoothed_image: np.ndarray) -> np.ndarray:
    threshold = random.randint(MIN_THRESHOLD, MAX_THRESHOLD)
    stretch = skimage.exposure.rescale_intensity(
        smoothed_image, in_range='image', out_range=(0,255),
    )
    stretch = stretch.astype(np.uint8)
    return cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]


def _smooth_shapes(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    if np.random.random() <= SMOOTHING_PROBABILITY:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _create_shape_stripes(image_dimensions: ImageDimensions) -> np.ndarray:
    mask = np.zeros((image_dimensions.height, image_dimensions.width), dtype=np.int32)
    num_stripes = random.randint(MIN_SHAPE_STRIPES, MAX_SHAPE_STRIPES)
    for _ in range(num_stripes):
        thickness = random.randint(MIN_SHAPE_STRIPE_THICKNESS, MAX_SHAPE_STRIPE_THICKNESS)
        _create_stripe(image_dimensions, mask, thickness, fill_value=255)
    return mask


def _create_stripe(
        image_dimensions: ImageDimensions, mask: np.ndarray, thickness: int, fill_value: int,
) -> None:
    start_point_x = random.randint(0, image_dimensions.width)
    start_point_y = random.randint(0, image_dimensions.height)
    start_point = (start_point_y, start_point_x)
    end_point_x = random.randint(0, image_dimensions.width)
    end_point_y = random.randint(0, image_dimensions.height)
    end_point = (end_point_y, end_point_x)
    cv2.line(mask, start_point, end_point, fill_value, thickness)


def _is_volume_fraction_exceeded(
        phase_mask: np.ndarray, total_mask_pixels: int, volume_fraction: float,
) -> bool:
    total_mask_entries = phase_mask.sum()
    return  total_mask_entries / total_mask_pixels > volume_fraction


def _get_phase_2_mask(phase_1_mask: np.ndarray) -> np.ndarray:
    return np.where(phase_1_mask == 0, 1, 0)


def _augment_two_phase_image(
        two_phase_image: np.ndarray,
        noise_augmentation: bool | float,
        alpha_augmentation: bool | float,
        anomaly_patch_augmentation: bool | float,
        stripe_augmentation: bool | float,
        phase_1_mask: np.ndarray,
        phase_2_mask: np.ndarray,
        image_dimensions: ImageDimensions,
) -> np.ndarray:
    if _bool_or_random_check(noise_augmentation):
        two_phase_image = _apply_noise_augmentation_to_phase(
            two_phase_image, image_dimensions, phase_1_mask,
        )
        two_phase_image = _apply_noise_augmentation_to_phase(
            two_phase_image, image_dimensions, phase_2_mask,
        )
    if _bool_or_random_check(alpha_augmentation):
        two_phase_image = _apply_alpha_augmentation(two_phase_image, image_dimensions)
    if _bool_or_random_check(anomaly_patch_augmentation):
        two_phase_image = _apply_anomaly_patch_augmentation(two_phase_image, image_dimensions)
    if _bool_or_random_check(stripe_augmentation):
        two_phase_image = _apply_stripe_augmentation(two_phase_image, image_dimensions)
    return two_phase_image


def _apply_noise_augmentation_to_phase(
        two_phase_image: np.ndarray, image_dimensions: ImageDimensions, phase_mask: np.ndarray
) -> np.ndarray:
    image_shape = (image_dimensions.height, image_dimensions.width)
    noise_std = np.random.random() * MAX_NOISE_STD
    noise = np.random.normal(0, noise_std, image_shape)
    two_phase_image += (phase_mask * noise).astype(int)
    return two_phase_image


def _apply_alpha_augmentation(image: np.ndarray, image_dimensions: ImageDimensions) -> np.ndarray:
    num_patches = random.randint(1, MAX_RAND_ALPHA_PATCHES)
    for _ in range(num_patches):
        patch_width = random.randint(1, MAX_ALPHA_PATCH_DIM)
        patch_height = random.randint(1, MAX_ALPHA_PATCH_DIM)
        patch_x = random.randint(0, image_dimensions.width - patch_width)
        patch_y = random.randint(0, image_dimensions.height - patch_height)
        random_alpha_value_diff = random.randint(5, 20)
        image[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] -= random_alpha_value_diff
    return image


def _apply_anomaly_patch_augmentation(
        image: np.ndarray, image_dimensions: ImageDimensions,
) -> np.ndarray:
    anomaly_patches = _create_anomaly_augmentation(image_dimensions)
    mask = anomaly_patches.astype(bool)
    image[mask] = anomaly_patches[mask]
    return image


def _create_anomaly_augmentation(image_dimensions: ImageDimensions) -> np.ndarray:
    mask = np.zeros((image_dimensions.height, image_dimensions.width), dtype=np.int32)
    num_anomaly_patches = random.randint(MIN_ANOMALY_PATCHES, MAX_ANOMALY_PATCHES)
    for _ in range(num_anomaly_patches):
        selected_shape = random.choice(ANOMALY_PATCH_SHAPES)
        fill_value = random.choice([1, 255])
        if selected_shape == 'rectangle':
            _generate_rectangle(mask, image_dimensions, fill_value)
        elif selected_shape == 'circle':
            _generate_cirle(mask, image_dimensions, fill_value)
    return mask


def _generate_rectangle(
        mask: np.ndarray, image_dimensions: ImageDimensions, fill_value: int,
) -> None:
    upper_left_x = random.randint(0, image_dimensions.width - MAX_ANOMALY_RECT_DIM)
    upper_left_y = random.randint(0, image_dimensions.height - MAX_ANOMALY_RECT_DIM)
    upper_left_corner = (upper_left_x, upper_left_y)
    lower_right_x = random.randint(upper_left_x + 1, upper_left_x + MAX_ANOMALY_RECT_DIM)
    lower_right_y = random.randint(upper_left_y + 1, upper_left_y + MAX_ANOMALY_RECT_DIM)
    lower_right_corner = (lower_right_x, lower_right_y)
    cv2.rectangle(mask, upper_left_corner, lower_right_corner, fill_value, thickness=-1)


def _generate_cirle(
        mask: np.ndarray, image_dimensions: ImageDimensions, fill_value: int,
) -> None:
    radius = random.randint(1, MAX_ANOMALY_CIRCLE_RADIUS)
    center_x = random.randint(0, image_dimensions.width - radius)
    center_y = random.randint(0, image_dimensions.height - radius)
    center = (center_x, center_y)
    cv2.circle(mask, center, radius, fill_value, thickness=-1)


def _apply_stripe_augmentation(image: np.ndarray, image_dimensions: ImageDimensions) -> np.ndarray:
    stripes = np.zeros((image_dimensions.height, image_dimensions.width), dtype=np.int32)
    num_stripes = random.randint(MIN_AUGMENT_STRIPES, MAX_AUGMENT_STRIPES)
    for _ in range(num_stripes):
        thickness = random.randint(MIN_AUGMENT_STRIPE_THICKNESS, MAX_AUGMENT_STRIPE_THICKNESS)
        fill_value = random.choice([1, 225])
        _create_stripe(image_dimensions, stripes, thickness, fill_value)
    mask = stripes.astype(bool)
    image[mask] = stripes[mask]
    return image


def _get_texture_volume_fractions(
        two_phase_image: np.ndarray,
        phase_1_mask: np.ndarray,
        phase_2_mask: np.ndarray,
        image_dimensions: ImageDimensions,
) -> tuple[TextureVolFrac, TextureVolFrac]:
    texture_vol_frac_1 = _get_single_texture_volume_fraction(
        two_phase_image, phase_1_mask, image_dimensions,
    )
    texture_vol_frac_2 = _get_single_texture_volume_fraction(
        two_phase_image, phase_2_mask, image_dimensions,
    )
    texture_vol_fracs = [texture_vol_frac_1, texture_vol_frac_2]
    random.shuffle(texture_vol_fracs)
    return texture_vol_fracs


def _get_single_texture_volume_fraction(
        two_phase_image: np.ndarray, mask: np.ndarray, image_dimensions: ImageDimensions,
) -> TextureVolFrac:
    volume_fraction = mask.sum() / (image_dimensions.height * image_dimensions.width)
    texture_image = _get_texture_image(two_phase_image, mask, image_dimensions)
    return TextureVolFrac(texture_image, volume_fraction)


def _get_texture_image(
        two_phase_image: np.ndarray, mask: np.ndarray, image_dimensions: ImageDimensions,
) -> torch.Tensor:
    texture_sample_bb = _get_texture_sample_bounding_box(mask, image_dimensions)
    texture_image = two_phase_image[
        texture_sample_bb[0]:texture_sample_bb[0]+texture_sample_bb[2],
        texture_sample_bb[1]:texture_sample_bb[1]+texture_sample_bb[3],
    ]
    return torch.as_tensor(texture_image, dtype=DTYPE)


def _get_texture_sample_bounding_box(
        mask: np.ndarray, image_dimensions: ImageDimensions,
) -> tuple[int]:
    max_bb = (0, 0, 0, 0)
    for height_index in range(image_dimensions.height):
        max_height = _get_bounding_box_max_height(image_dimensions, height_index)
        for width_index in range(image_dimensions.width):
            if not mask[height_index, width_index]:
                continue
            max_width = _get_bounding_box_max_width(image_dimensions, width_index)
            if max_width * max_height <= max_bb[2] * max_bb[3]:
                continue
            max_bb = _find_best_bounding_box_dimensions(
                mask, max_bb, max_height, height_index, max_width, width_index,
            )
            if max_bb[2] * max_bb[3] >= MAX_TEXTURE_SAMPLE_SIZE:
                return max_bb
    return max_bb


def _get_bounding_box_max_height(image_dimensions: ImageDimensions, height_index: int) -> int:
    maximum_possible_image_height = image_dimensions.height - (height_index + 1)
    if maximum_possible_image_height > MAX_TEXTURE_SAMPLE_DIM:
        return MAX_TEXTURE_SAMPLE_DIM
    return maximum_possible_image_height


def _get_bounding_box_max_width(image_dimensions: ImageDimensions, width_index: int) -> int:
    maximum_possible_image_width = image_dimensions.width - (width_index + 1)
    if maximum_possible_image_width > MAX_TEXTURE_SAMPLE_DIM:
        return MAX_TEXTURE_SAMPLE_DIM
    return maximum_possible_image_width


def _find_best_bounding_box_dimensions(
        mask: np.ndarray,
        max_bb: tuple[int],
        max_height: int,
        height_index: int,
        max_width: int,
        width_index: int,
) -> tuple[int]:
    for bb_height in range(1, max_height + 1):
        if not mask[height_index + bb_height, width_index]: # break the loop at the max height
            break
        for bb_width in range(1, max_width + 1):
            if not mask[height_index + bb_height, width_index + bb_width]:
                max_width = bb_width
                break
            if max_bb[2] * max_bb[3] < bb_height * bb_width:
                max_bb = (height_index, width_index, bb_height, bb_width)
            if max_bb[2] * max_bb[3] >= MAX_TEXTURE_SAMPLE_SIZE:
                return max_bb
        if bb_width == 1 and bb_height == 1:
            break
    return max_bb


def _bool_or_random_check(check_value: bool | float) -> bool:
    return (isinstance(check_value, bool) and check_value) or \
        (isinstance(check_value, float) and check_value < np.random.random())
