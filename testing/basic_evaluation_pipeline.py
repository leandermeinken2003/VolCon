"""Analyze a testrun by basic metrics."""

import glob
from typing import Callable

import torch
import pandas as pd

from model.volume_fraction_model import VolumeFractionModel
from utils.postprocessing import (
    one_stage_post_processing,
    two_stage_post_processing,
)


TESTRUN_ID = "5.7M_Context_Linear1"
TESTRUN_PATH = f'C:\\Users\\leand\\CES\\Semester VI\\Bachelorarbeit\\testruns\\{TESTRUN_ID}\\'
MODEL_PATH = TESTRUN_PATH + 'model.pth'

TESTDATA_PATH = 'testing/processed_testing_data/'

ANALYSIS_PATH = TESTRUN_PATH + 'analysis/'
GOOD_THRESHOLD = 0.1
BAD_THRESHOLD = 0.2


def main():
    """Analyze a testrun."""
    testcases_data = _load_testcases_data()
    testcases_data = _add_predicted_data(testcases_data)
    testcases_data = _postprocess_predictions(testcases_data)
    _analyse_general_metrics(testcases_data)
    _get_best_scoring_images(testcases_data)
    _get_worst_scoring_images(testcases_data)


def _load_testcases_data() -> pd.DataFrame:
    testcase_paths = glob.glob(TESTDATA_PATH + '*')
    testcases_data = []
    for testcase_path in testcase_paths:
        testcase_id = testcase_path.split('/')[-1]
        testcase_id = testcase_id.split('\\')[-1]
        testcase_data = pd.read_csv(testcase_path + '/volume_fractions.csv')
        testcase_data['testcase_id'] = testcase_id
        testcases_data.append(testcase_data)
    return pd.concat(testcases_data, axis=0)


def _add_predicted_data(testcases_data: pd.DataFrame) -> pd.DataFrame:
    model = torch.load(MODEL_PATH).to("cuda")
    model.eval()
    new_testcases_data = []
    for testcase_id, testcase_data in testcases_data.groupby('testcase_id'):
        testcase_data = _add_prediction_to_testcase(testcase_id, testcase_data, model)
        new_testcases_data.append(testcase_data)
    new_testcases_data = pd.concat(new_testcases_data, axis=0)
    return new_testcases_data


def _add_prediction_to_testcase(
        testcase_id: str, testcase_data: pd.DataFrame, model: VolumeFractionModel,
) -> pd.DataFrame:
    testcase_path = TESTDATA_PATH + testcase_id
    microstructure = torch.load(
        testcase_path + '/microstructure.pt', weights_only=True,
    ).to("cuda") / 255
    material_1_texture = torch.load(
        testcase_path + '/material_1_sample.pt', weights_only=True,
    ).to("cuda") / 255
    material_2_texture = torch.load(
        testcase_path + '/material_2_sample.pt', weights_only=True,
    ).to("cuda") / 255
    with torch.no_grad():
        volume_fraction_1 = model(microstructure, material_1_texture, material_2_texture)
        volume_fraction_2 = model(microstructure, material_2_texture, material_1_texture)
        testcase_data['predicted_volume_fraction'] = [
            volume_fraction_1.cpu().numpy()[0][0], volume_fraction_2.cpu().numpy()[0][0],
        ]
    return testcase_data


def _postprocess_predictions(testcases_data: pd.DataFrame) -> pd.DataFrame:
    new_testcases_data = []
    for _, testcase_data in testcases_data.groupby('testcase_id'):
        volume_fraction_predictions = testcase_data['predicted_volume_fraction'].tolist()
        testcase_data['one_stage_post_processed_volume_fraction'] = \
            _calculate_postprocessed_volume_fractions(
                one_stage_post_processing, volume_fraction_predictions,
            )
        testcase_data['two_stage_post_processed_volume_fraction'] = \
            _calculate_postprocessed_volume_fractions(
                two_stage_post_processing, volume_fraction_predictions,
            )
        new_testcases_data.append(testcase_data)
    return pd.concat(new_testcases_data, axis=0).reset_index()


def _calculate_postprocessed_volume_fractions(
        postprocessing_func: Callable, volume_fraction_predictions: list[float, float],
) -> list[float, float]:
    volume_fraction_prediction_1 = volume_fraction_predictions[0]
    volume_fraction_prediction_2 = volume_fraction_predictions[1]
    return postprocessing_func(volume_fraction_prediction_1, volume_fraction_prediction_2)


def _analyse_general_metrics(testcases_data: pd.DataFrame) -> None:
    testcases_data['absolute_error'], _ = _output_metrics_for_predictions(
        testcases_data, 'predicted_volume_fraction',
    )
    _output_metrics_for_predictions(testcases_data, 'one_stage_post_processed_volume_fraction')
    _output_metrics_for_predictions(testcases_data, 'two_stage_post_processed_volume_fraction')


def _output_metrics_for_predictions(testcases_data: pd.DataFrame, prediction_name: str) -> None:
    absolute_error = _absolute_error(testcases_data, prediction_name)
    percentage_error = _percentage_error(testcases_data, prediction_name)
    print(f"Prediction name: {prediction_name}")
    print(f"mean ae: {absolute_error.mean()} | median ae: {absolute_error.median()} | std ae: {absolute_error.std()}")
    print(f"mean pe: {percentage_error.mean()} | median pe: {percentage_error.median()}")
    return absolute_error, percentage_error


def _absolute_error(testcases_data: pd.DataFrame, prediction_name: str) -> float:
    return abs(testcases_data['volume_fraction'] - testcases_data[prediction_name])


def _percentage_error(testcases_data: pd.DataFrame, prediction_name: str) -> float:
    return abs(testcases_data['volume_fraction'] - testcases_data[prediction_name]) /\
        testcases_data['volume_fraction']


def _get_best_scoring_images(testcases_data: pd.DataFrame) -> None:
    print("Best testcases:")
    for testcase_id, testcase_data in testcases_data.groupby("testcase_id"):
        if (testcase_data['absolute_error'] < GOOD_THRESHOLD).all():
            print(testcase_id)
            print(testcase_data)


def _get_worst_scoring_images(testcases_data: pd.DataFrame) -> None:
    print("Worst testcases:")
    for testcase_id, testcase_data in testcases_data.groupby("testcase_id"):
        if (testcase_data['absolute_error'] > BAD_THRESHOLD).any():
            print(testcase_id)
            print(testcase_data)


if __name__ == '__main__':
    main()
