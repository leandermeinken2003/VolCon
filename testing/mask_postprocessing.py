"""Postprocess the drawn masks into a coherent testing sample."""

import os
from glob import glob
from collections import namedtuple

import torch
import numpy as np
import pandas as pd
from PIL import Image


COMPOSITION_ID = "Bi70Sn30"
IMAGE_ID = "000763"
TESTCASE_ID = f"{COMPOSITION_ID}_{IMAGE_ID}"
COMPOSITION_DIR = "testing/unprocessed_testing_data/{COMPOSITION_ID}"
IMAGE_PATH = COMPOSITION_DIR + f"/images/{IMAGE_ID}.jpg"
MASKS_DIR = COMPOSITION_DIR + f"/masks/{IMAGE_ID}/"
TESTDATA_DIR = f"testing/processed_testing_data/{TESTCASE_ID}/"
MIN_IMAGE_SIZE = 256
MAX_IMAGE_SIZE = 384

InfoBoxDim = namedtuple("InfoBoxDim", ["height", "width"])

def main() -> None:
    """Postprocess the data for one image."""
    if not os.path.exists(TESTDATA_DIR):
        os.makedirs(TESTDATA_DIR)
    microstructure_image = Image.open(IMAGE_PATH)
    microstructure_image = microstructure_image.convert(mode="L")
    microstructure_image = _resize_image(microstructure_image)
    microstructure_image_array = np.array(microstructure_image)
    mask_image_paths = _get_image_mask_paths()
    info_box_dim = _get_info_box_dim(mask_image_paths)
    cropped_microstructure_array = _save_cropped_microstructure_image(
        microstructure_image_array, info_box_dim,
    )
    _save_texture_images(cropped_microstructure_array, info_box_dim, mask_image_paths)


def _resize_image(image: Image.Image) -> Image.Image:
    image_height = image.height
    image_width = image.width
    if image_height < MIN_IMAGE_SIZE and image_height <= image_width:
        scaling_factor = MIN_IMAGE_SIZE / image_height
    elif image_width < MIN_IMAGE_SIZE and image_width < image_height:
        scaling_factor = MIN_IMAGE_SIZE / image_width
    elif image_height > MAX_IMAGE_SIZE and image_height >= image_width:
        scaling_factor = MAX_IMAGE_SIZE / image_height
    elif image_width > MAX_IMAGE_SIZE and image_width > image_height:
        scaling_factor = MAX_IMAGE_SIZE / image_width
    else:
        scaling_factor = 1
    return image.resize(
        (int(scaling_factor * image_width), int(scaling_factor * image_height)),
    )


def _get_image_mask_paths() -> list[str]:
    all_mask_paths = glob(MASKS_DIR + "*")
    return [
        mask_path.replace("\\", "/") for mask_path in all_mask_paths if mask_path.endswith(".png")
    ]


def _get_info_box_dim(mask_image_path: list[str]) -> InfoBoxDim:
    info_box_mask_path = [path for path in mask_image_path if "info_box" in path][0]
    info_box_mask = _get_mask_from_image_path(info_box_mask_path)
    width = _get_info_box_width(info_box_mask)
    height = _get_info_box_height(info_box_mask)
    if width < info_box_mask.shape[0] - height:
        info_box_dim = InfoBoxDim(info_box_mask.shape[0], width)
    else:
        info_box_dim = InfoBoxDim(height, 0)
    return info_box_dim


def _get_mask_from_image_path(image_path: str) -> np.ndarray:
    mask_image = Image.open(image_path)
    mask_image = mask_image.convert("L")
    mask_image = _resize_image(mask_image)
    mask = np.array(mask_image)
    return np.where(mask > 0, 0, 1)


def _get_info_box_width(info_box_mask: np.ndarray) -> int:
    mask_width = info_box_mask.shape[1]
    for width_x in range(mask_width):
        if not np.sum(info_box_mask[:, width_x]):
            return width_x
    return mask_width


def _get_info_box_height(info_box_mask: np.ndarray) -> int:
    mask_height = info_box_mask.shape[0]
    for height_x in range(mask_height):
        if not np.sum(info_box_mask[mask_height - 1 - height_x, :]):
            return mask_height - 1 - height_x
    return mask_height


def _save_cropped_microstructure_image(
        microstructure_image: np.ndarray, info_box_dim: InfoBoxDim,
) -> None:
    info_box_height = info_box_dim.height
    info_box_width = info_box_dim.width
    cropped_microstructure_image = microstructure_image[:info_box_height, info_box_width:]
    _save_image_array_as_tensor_file(
        cropped_microstructure_image, TESTDATA_DIR + "/microstructure.pt",
    )
    Image.fromarray(cropped_microstructure_image).show()
    return cropped_microstructure_image


def _save_image_array_as_tensor_file(array: np.ndarray, path: str) -> None:
    tensor = torch.as_tensor(array)
    tensor = tensor.unsqueeze(0).unsqueeze(0).type(torch.float32)
    torch.save(tensor, path)

def _save_texture_images(
        cropped_microstructure_array: np.ndarray,
        info_box_dim: InfoBoxDim,
        mask_image_paths: list[str],
) -> None:
    texture_names = _get_texture_names(mask_image_paths)
    _save_volume_fractions(info_box_dim, texture_names)
    _save_texture_samples(
        cropped_microstructure_array, info_box_dim, texture_names,
    )


def _get_texture_names(mask_image_path: list[str]) -> list[str]:
    texture_image_array_paths = [
        path.split("/")[-1] for path in mask_image_path if path.endswith("_texture_image.png")
    ]
    return [path.split("_texture_image")[0] for path in texture_image_array_paths]


def _save_volume_fractions(info_box_dim: InfoBoxDim, texture_names: list[str]) -> None:
    volume_fraction_dict = {
        "texture_name": [],
        "volume_fraction": [],
    }
    info_box_height = info_box_dim.height
    info_box_width = info_box_dim.width
    for texture_name in texture_names[:-1]:
        texture_mask_path = MASKS_DIR + f"{texture_name}_mask.png"
        texture_mask = _get_mask_from_image_path(texture_mask_path)
        cropped_texture_mask = texture_mask[:info_box_height, info_box_width:]
        volume_fraction_dict["texture_name"].append(texture_name)
        image_size = int(cropped_texture_mask.shape[0] * cropped_texture_mask.shape[1])
        mask_count = int(cropped_texture_mask.sum())
        volume_fraction_dict["volume_fraction"].append(mask_count / image_size)
    _get_final_texture_volume_fraction(texture_names[-1], volume_fraction_dict)
    volume_fraction_df = pd.DataFrame.from_dict(volume_fraction_dict)
    volume_fraction_df.to_csv(TESTDATA_DIR + "volume_fractions.csv")


def _get_final_texture_volume_fraction(
        texture_name: str, volume_fraction_dict: dict[str, list],
) -> None:
    volume_fraction_dict["texture_name"].append(texture_name)
    volume_fraction_dict["volume_fraction"].append(
        1 - sum(volume_fraction_dict["volume_fraction"])
    )


def _save_texture_samples(
        cropped_microstructure_array: np.ndarray,
        info_box_dim: InfoBoxDim,
        texture_names: list[str],
    ) -> None:
    info_box_height = info_box_dim.height
    info_box_width = info_box_dim.width
    for texture_name in texture_names:
        texture_sample_mask_path = MASKS_DIR + f"{texture_name}_texture_image.png"
        texture_sample_mask = _get_mask_from_image_path(texture_sample_mask_path)
        cropped_texture_sample_mask = texture_sample_mask[:info_box_height, info_box_width:]
        texture_sample = _get_texture_sample(
            cropped_microstructure_array, cropped_texture_sample_mask,
        )
        Image.fromarray(texture_sample).show()
        _save_image_array_as_tensor_file(
            texture_sample, TESTDATA_DIR + f"/{texture_name}_sample.pt",
        )


def _get_texture_sample(
        cropped_microstructure_array: np.ndarray, cropped_texture_sample_mask: np.ndarray,
) -> np.ndarray:
    height = cropped_microstructure_array.shape[0]
    width = cropped_microstructure_array.shape[1]
    texture_sample = []
    for y in range(height):
        texture_sample_row = []
        for x in range(width):
            if cropped_texture_sample_mask[y, x]:
                texture_sample_row.append(cropped_microstructure_array[y, x])
        if texture_sample_row:
            texture_sample.append(texture_sample_row)
    return np.array(texture_sample)


if __name__ == '__main__':
    main()
