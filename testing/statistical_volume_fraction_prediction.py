"""
Implementation of statistical method for predicting volume fractions.

This implementation is described/based on the paper:
A.W Wilson, J.D Madison, G Spanos (2001). Determining phase volume fraction in steels by
electron backscattered diffraction. Scripta Materialia Volume 45, Issue 12, Pages 1335-1340.
https://www.sciencedirect.com/science/article/pii/S135964620101137X

Instead of using a histogram and finding the peaks however just the average pixel value is used
as a threshold, as the micrographs are not well suited to the exact implementation of the method.
"""

import glob

import pandas as pd
import torch


TESTDATA_PATH = 'testing/processed_testing_data/'

N_BINS = 20


def main() -> None:
    """Evaluate the performance of the statistical prediction method."""
    testcases_data = _load_testcases_data()
    testcases_data = _add_predicted_data(testcases_data)
    _evaluate_performance(testcases_data)


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
    new_testcases_data = []
    for testcase_id, testcase_data in testcases_data.groupby('testcase_id'):
        testcase_data = _add_prediction_to_testcase(testcase_id, testcase_data)
        new_testcases_data.append(testcase_data)
    new_testcases_data = pd.concat(new_testcases_data, axis=0)
    return new_testcases_data


def _add_prediction_to_testcase(testcase_id: str, testcase_data: pd.DataFrame) -> pd.DataFrame:
    testcase_path = TESTDATA_PATH + testcase_id
    microstructure = torch.load(testcase_path + '/microstructure.pt', weights_only=True)
    threshold = torch.mean(microstructure)
    total_pixels = microstructure.shape[2] * microstructure.shape[3]
    volume_fraction_1 = ((microstructure > threshold).sum() / total_pixels).item()
    volume_fraction_2 = ((microstructure <= threshold).sum() / total_pixels).item()
    if _is_first_texture_lighter(testcase_path):
        testcase_data['predicted_volume_fraction'] = [volume_fraction_1, volume_fraction_2]
    else:
        testcase_data['predicted_volume_fraction'] = [volume_fraction_2, volume_fraction_1]
    return testcase_data


def _is_first_texture_lighter(testcase_path: str) -> bool:
    material_1_texture = torch.load(testcase_path + '/material_1_sample.pt', weights_only=True)
    material_2_texture = torch.load(testcase_path + '/material_2_sample.pt', weights_only=True)
    material_1_average = material_1_texture.mean()
    material_2_average = material_2_texture.mean()
    return material_1_average > material_2_average


def _evaluate_performance(testcases_data: pd.DataFrame) -> None:
    absolute_error = abs(
        testcases_data['volume_fraction'] - testcases_data['predicted_volume_fraction']
    )
    print('mean absolute error: ', absolute_error.mean())
    print('median absolute error: ', absolute_error.median())
    print('std absolute error: ', absolute_error.std())


if __name__ == '__main__':
    main()
