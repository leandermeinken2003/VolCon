"""Generate an image with two phases, that have different textures."""
#pylint: disable=no-member, import-error, too-many-arguments, too-many-locals

from math import (
    floor,
    ceil,
)
import random
from collections import namedtuple


import cv2
import torch
import numpy as np
from numpy.random import default_rng
import skimage


DTYPE = torch.float32

MAX_ITER = 20

MAX_SHAPES = 20
MIN_SHAPES = 2
#generic shapes hyperparameters
MAX_GENERIC_SHAPE_DIM = 40
#gaussian smoothed hyperparameters
MIN_THRESHOLD = 200
SMOOTHING_PROBABILITY = 0.5
#stripes hyperparameters
MAX_SHAPE_STRIPE_THICKNESS = 10
MIN_SHAPE_STRIPE_THICKNESS = 2

MAX_GRID_SIZE = 128
MAX_INTERVAL=10
MAX_MAX_ABS_ANGLE=45

MAX_NOISE_STD = 20

MAX_RAND_ALPHA_PATCHES = 5
MAX_ALPHA_PATCH_DIM = 100

MAX_ANOMALY_PATCH_DIM = 10

MAX_AUGMENT_STRIPE_THICKNESS = 2
MIN_AUGMENT_STRIPE_THICKNESS = 1

MAX_TEXTURE_SAMPLE_DIM = 25
MAX_TEXTURE_SAMPLE_SIZE = 400


ImageDimensions = namedtuple("ImageDimensions", ["image_height", "image_width"])
TextureVolFrac = namedtuple("TextureMask", ["texture_image", "volume_fraction"])


def generate_two_phase_image(
        image_dimensions: ImageDimensions,
        shapes_type: str | list[str] = "generic",
        combine_shape_types: bool | float = 0.5,
        min_phase_distance: int = MAX_NOISE_STD,
        alpha_augmentation: bool | float = 0.5,
        noise_augmentation: bool | float = 0.5,
        anomaly_patch_augmentation: bool | float = 0.5,
        stripe_augmentation: bool | float = 0.5,
    ) -> tuple[torch.Tensor, TextureVolFrac, TextureVolFrac]:
    """Generate a two phase image and apply the selected augmentations.
    
    :param shapes_type: Decide the kind of shapes that can be used in the two phase image. The
    type of shapes that can be used are generic, gaussian_smoothed or stripe.
    :param combine_shape_types:Decide, or give percentage probability of use, wether to combine 
    different types of shapes can be combined in one image.
    :param min_phase_distance: The minimum color distance between the different phases.
    :param alpha_augmentation: Decide, or give percentage probability of use, wether to use alpha 
    image augmentation, meaning that random area of the image are set to lower alpha values.
    :param noise_augmentation: Decide, or give percentage probability of use, if to use noise 
    augmentation. In noise augmentation noise of a random intensity is added to the different
    phases.
    :param anomaly_patch_augmentation: Decide, or give percentage probability of use, wether to use 
    anomaly-patch augmentation. In anomaly-patch augmentation random patches of different colors are
    added to both phases.
    :param stripe_augmentation: Decide, or give percentage probability of use, wether to use stripe
    augmentation. In stripe augmentation, stripes of different colors are added, which go across the
    entire image.
    """
    if isinstance(shapes_type, str):
        shapes_type = [shapes_type]
    if not _bool_or_random_check(combine_shape_types):
        selected_shape_type = random.choice(shapes_type)
        shapes_type = [selected_shape_type]
    phase_1_color, phase_2_color = _get_phase_colors(min_phase_distance)
    phase_1_mask = _get_phase_1_mask(shapes_type, image_dimensions)
    phase_2_mask = _get_phase_2_mask(phase_1_mask)
    two_phase_image = _create_two_phase_image(
        phase_1_color, phase_1_mask, phase_2_color, phase_2_mask, noise_augmentation,
    )
    two_phase_image = _augment_two_phase_image(
        two_phase_image,
        alpha_augmentation,
        anomaly_patch_augmentation,
        stripe_augmentation,
        image_dimensions,
    )
    texture_vol_frac_1, texture_vol_frac_2 = _get_texture_volume_fractions(
        two_phase_image, phase_1_mask, phase_2_mask, image_dimensions,
    )
    two_phase_image = torch.as_tensor(
        two_phase_image.reshape((1, image_dimensions.image_height, image_dimensions.image_width)),
        dtype=DTYPE,
    )
    return two_phase_image, texture_vol_frac_1, texture_vol_frac_2


def _get_phase_colors(min_phase_distance: int) -> tuple[int, int]:
    """Get the color of the phases."""
    assert min_phase_distance >= 0, "The min_phase_distance has to be a positive value."
    phase_1_color = np.random.randint(min_phase_distance + 1, 255)
    phase_2_color = np.random.randint(0, phase_1_color - min_phase_distance)
    phase_colors = [phase_1_color, phase_2_color]
    random.shuffle(phase_colors)
    return phase_colors


def _get_phase_1_mask(shapes_types: list[str], image_dimensions: ImageDimensions) -> np.ndarray:
    volume_fraction_phase_1 = np.random.random() * 0.90 + 0.05
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    phase_1_mask = np.zeros((image_height, image_width), dtype=np.int32)
    total_mask_entries = image_height * image_width
    iter_counter = 0
    while phase_1_mask.sum() / total_mask_entries < volume_fraction_phase_1 and \
        iter_counter < MAX_ITER:
        shape_type = random.choice(shapes_types)
        if shape_type == "generic":
            phase_1_mask += _create_generic_shapes(image_height, image_width, MAX_GENERIC_SHAPE_DIM)
        elif shape_type == "gaussian_smoothed":
            threshold = np.random.randint(MIN_THRESHOLD, 245)
            phase_1_mask += _create_gaussian_smoothed_shapes(image_height, image_width, threshold)
        elif shape_type == "stripe":
            phase_1_mask += _create_stripes(
                image_height, image_width, MAX_SHAPE_STRIPE_THICKNESS, MIN_SHAPE_STRIPE_THICKNESS,
            )
        elif shape_type == "grid":
            phase_1_mask += _create_grid(image_height, image_width)
        phase_1_mask = np.where(phase_1_mask > 0, 1, 0)
        iter_counter += 1
    return phase_1_mask


def _create_generic_shapes(
        image_height: int, image_width: int, max_dimension: int, fill_value: int | list[int] = 255,
) -> np.ndarray:
    if isinstance(fill_value, int):
        fill_value = [fill_value]
    mask = np.zeros((image_height, image_width), dtype=np.int32)
    generic_shapes = ['rectangle', 'circle']
    thickness = -1 #Fill the shapes
    num_shapes = np.random.randint(MIN_SHAPES, MAX_SHAPES)
    for _ in range(num_shapes):
        selected_shape = random.choice(generic_shapes)
        if selected_shape == 'rectangle':
            upper_left_corner = (
                np.random.randint(0, image_width - max_dimension),
                np.random.randint(0, image_height - max_dimension),
            )
            lower_right_corner = (
                np.random.randint(
                    upper_left_corner[0] + 2,upper_left_corner[0] + max_dimension,
                ),
                np.random.randint(
                    upper_left_corner[1] + 1, upper_left_corner[1] + max_dimension,
                ),
            )
            current_fill_value = random.choice(fill_value)
            cv2.rectangle(
                mask, upper_left_corner, lower_right_corner, current_fill_value, thickness,
            )
        elif selected_shape == 'circle':
            max_radius = max_dimension // 2
            center = (
                np.random.randint(0, image_width - max_radius),
                np.random.randint(0, image_height - max_radius),
            )
            radius = np.random.randint(1, max_radius)
            current_fill_value = random.choice(fill_value)
            cv2.circle(mask, center, radius, current_fill_value, thickness)
    return mask


def _create_gaussian_smoothed_shapes(
        image_height: int,
        image_width: int,
        threshold: int,
) -> np.ndarray:
    rng = default_rng()
    noise = rng.integers(0, 255, (image_height, image_width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(
        blur, in_range='image', out_range=(0,255),
    ).astype(np.uint8)
    mask = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    if np.random.random() <= SMOOTHING_PROBABILITY:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _create_stripes(
        image_height: int,
        image_width: int,
        max_stripe_thickness: int,
        min_stripe_thickness: int,
        fill_value: int | list[int] = 255,
) -> np.ndarray:
    if isinstance(fill_value, int):
        fill_value = [fill_value]
    mask = np.zeros((image_height, image_width), dtype=np.int32)
    num_stripes = np.random.randint(MIN_SHAPES, MAX_SHAPES)
    for _ in range(num_stripes):
        thickness = np.random.randint(min_stripe_thickness, max_stripe_thickness)
        start_point = (np.random.randint(0, image_height), np.random.randint(0, image_width))
        end_point = (np.random.randint(0, image_height), np.random.randint(0, image_width))
        current_fill_value = random.choice(fill_value)
        cv2.line(mask, start_point, end_point, current_fill_value, thickness)
    return mask


def _create_grid(image_height: int, image_width: int) -> np.ndarray:
    max_stripe_width = np.random.randint(2, 8)
    grid_height = np.random.randint(MAX_INTERVAL + max_stripe_width, MAX_GRID_SIZE)
    grid_width = np.random.randint(MAX_INTERVAL + max_stripe_width, MAX_GRID_SIZE)
    x_interval = np.random.randint(1, MAX_INTERVAL)
    y_interval = np.random.randint(1, MAX_INTERVAL)
    max_abs_angle = np.deg2rad(np.random.randint(0, MAX_MAX_ABS_ANGLE))
    start_position_x, start_position_y = _get_grid_start_position(
        image_height, image_width, grid_height, grid_width, max_abs_angle,
    )
    mask = np.zeros((image_height, image_width), dtype=np.int32)
    x_position = np.random.randint(0, x_interval)
    while x_position < grid_width:
        stripe_width = np.random.randint(1, max_stripe_width)
        angle = (np.random.random() - 0.5) * 2 * max_abs_angle
        current_start_x = start_position_x + x_position + ceil(stripe_width / 2)
        cv2.line(
            mask,
            (current_start_x, start_position_y),
            (floor(current_start_x + np.sin(angle) * grid_width), floor(start_position_y + np.cos(angle) * grid_width)),
            color=1,
            thickness=stripe_width,
        )
        x_position += stripe_width + x_interval
    y_position = np.random.randint(0, y_interval)
    while y_position < grid_height:
        stripe_width = np.random.randint(1, max_stripe_width)
        angle = (np.random.random() - 0.5) * 2 * max_abs_angle
        current_start_y = start_position_y + y_position + ceil(stripe_width / 2)
        cv2.line(
            mask,
            (start_position_x, current_start_y),
            (floor(start_position_x + np.cos(angle) * grid_height), floor(current_start_y + np.sin(angle) * grid_height)),
            color=1,
            thickness=stripe_width,
        )
        y_position += stripe_width + y_interval
    return mask


def _get_grid_start_position(
        image_height: int,
        image_width: int,
        grid_height: int,
        grid_width: int,
        max_abs_angle: float,
) -> tuple[int, int]:
    start_left = np.random.choice([True, False])
    if start_left:
        start_position_x = np.random.randint(
            0, np.ceil(image_width - np.sin(max_abs_angle) * grid_height) - 1,
        )
    else:
        min_x = np.ceil(np.sin(max_abs_angle) * grid_height) - 1
        min_x = min_x if min_x >= 0 else 0
        start_position_x = np.random.randint(min_x, image_width - 1)
    start_top = np.random.choice([True, False])
    if start_top:
        start_position_y = np.random.randint(
            0, np.ceil(image_height - np.sin(max_abs_angle) * grid_width) - 1,
        )
    else:
        min_y = np.ceil(np.sin(max_abs_angle) * grid_width) - 1
        min_y = min_y if min_y >= 0 else 0
        start_position_y = np.random.randint(min_y, image_height - 1)
    return start_position_x, start_position_y


def _get_phase_2_mask(phase_1_mask: np.ndarray) -> np.ndarray:
    return np.where(phase_1_mask == 0, 1, 0)


def _create_two_phase_image(
        phase_1_color: int,
        phase_1_mask: np.ndarray,
        phase_2_color: int,
        phase_2_mask: np.ndarray,
        noise_augmentation: bool | float,
) -> np.ndarray:
    if isinstance(noise_augmentation, float):
        noise_augmentation = np.random.random() < noise_augmentation
    image_part_phase_1 = _create_image_phase_part(phase_1_color, phase_1_mask, noise_augmentation)
    image_part_phase_2 = _create_image_phase_part(phase_2_color, phase_2_mask, noise_augmentation)
    return image_part_phase_1 + image_part_phase_2


def _create_image_phase_part(
        phase_color: int, phase_mask: np.ndarray, noise_augmentation: bool | float,
) -> np.ndarray:
    image_shape = (phase_mask.shape[0], phase_mask.shape[1])
    phase_image = phase_color * phase_mask
    if _bool_or_random_check(noise_augmentation):
        noise_probability = np.random.random()
        probability_noise_mask = np.random.rand(*image_shape) < noise_probability
        noise_std = np.random.random() * MAX_NOISE_STD
        noise = np.random.normal(0, noise_std, image_shape)
        phase_image += (probability_noise_mask * phase_mask * noise).astype(int)
    return phase_image


def _augment_two_phase_image(
        two_phase_image: np.ndarray,
        alpha_augmentation: bool | float,
        anomaly_patch_augmentation: bool | float,
        stripe_augmentation: bool | float,
        image_dimensions: ImageDimensions,
) -> np.ndarray:
    if _bool_or_random_check(anomaly_patch_augmentation):
        two_phase_image = _apply_anomaly_patch_augmentation(two_phase_image, image_dimensions)
    if _bool_or_random_check(stripe_augmentation):
        two_phase_image = _apply_stripe_augmentation(two_phase_image, image_dimensions)
    if _bool_or_random_check(alpha_augmentation):
        two_phase_image = _apply_alpha_augmentation(two_phase_image, image_dimensions)
    return two_phase_image


def _apply_alpha_augmentation(image: np.ndarray, image_dimensions: ImageDimensions) -> np.ndarray:
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    num_patches = np.random.randint(1, MAX_RAND_ALPHA_PATCHES)
    for _ in range(num_patches):
        patch_width = np.random.randint(1, MAX_ALPHA_PATCH_DIM)
        patch_height = np.random.randint(1, MAX_ALPHA_PATCH_DIM)
        patch_x = np.random.randint(0, image_width - patch_width)
        patch_y = np.random.randint(0, image_height - patch_height)
        random_alpha_value_diff = np.random.randint(5, 20)
        image[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] -= random_alpha_value_diff
    return image


def _apply_anomaly_patch_augmentation(
        image: np.ndarray, image_dimensions: ImageDimensions,
) -> np.ndarray:
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    anomaly_patches = _create_generic_shapes(
        image_height, image_width, MAX_ANOMALY_PATCH_DIM, [1, 255],
    )
    mask = anomaly_patches.astype(bool)
    image[mask] = anomaly_patches[mask]
    return image


def _apply_stripe_augmentation(image: np.ndarray, image_dimensions: ImageDimensions) -> np.ndarray:
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    stripes = _create_stripes(
        image_height,
        image_width,
        MAX_AUGMENT_STRIPE_THICKNESS,
        MIN_AUGMENT_STRIPE_THICKNESS,
        [1, 255],
    )
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
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    volume_fraction = mask.sum() / (image_height * image_width)
    texture_sample_bb = _get_texture_sample_bounding_box(mask, image_dimensions)
    texture_image = two_phase_image[
        texture_sample_bb[0]:texture_sample_bb[0]+texture_sample_bb[2],
        texture_sample_bb[1]:texture_sample_bb[1]+texture_sample_bb[3]
    ]
    texture_image = torch.as_tensor(texture_image, dtype=DTYPE)
    return TextureVolFrac(texture_image, volume_fraction)

def _get_texture_sample_bounding_box(
        mask: np.ndarray, image_dimensions: ImageDimensions,
) -> tuple[int]:
    max_bb = (0, 0, 0, 0)
    image_height = image_dimensions.image_height
    image_width = image_dimensions.image_width
    for y in range(image_height):
        max_height = MAX_TEXTURE_SAMPLE_DIM if image_height - y - 1 > MAX_TEXTURE_SAMPLE_DIM \
            else image_height - y - 1
        for x in range(image_width):
            max_width = MAX_TEXTURE_SAMPLE_DIM if image_width - x - 1 > MAX_TEXTURE_SAMPLE_DIM \
                else image_width - x - 1
            for h in range(1, max_height + 1):
                for w in range(1, max_width + 1):
                    if not mask[y + h - 1, x + w - 1]:
                        max_width = w
                        break
                    elif w > MAX_TEXTURE_SAMPLE_DIM / 2:
                        max_width = w - 1
                        break
                    if max_bb[2] * max_bb[3] < h * w:
                        max_bb = (y, x, h, w)
                    if max_bb[2] * max_bb[3] >= MAX_TEXTURE_SAMPLE_SIZE:
                        return max_bb
                if w == 1:
                    max_height = h
                    break
                elif h > MAX_TEXTURE_SAMPLE_DIM / 2:
                    max_height = h - 1
                    break
    return max_bb


def _bool_or_random_check(check_value: bool | float) -> bool:
    return (isinstance(check_value, bool) and check_value) or \
        (isinstance(check_value, float) and check_value < np.random.random())
