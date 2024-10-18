# Testing
The testing folder is used to benchmark the VolCon model, with a basic statistical method being defined for reference in statistical_volume_fraction_prediction.py. The file basic_evaluation_pipeline.py is used to evaluate a volcon model. In order to benchmark the model a local model has to exist, which is saved in a testrun. Change the path as follows to locally evaluate a VolCon model:
```python
TESTRUN_ID = "Your_Testrun_id"
TESTRUN_PATH = f'/Path/To/testrun/{TESTRUN_ID}'
```
The data in the folder ./processed_testing_data can be used to in general benchmark a VolCon model. In the case that additional data is to be added the data should be stored in the ./unprocessed_testing_data folder. In the next step the mask_creation_app.py is used to create the masks, the required masks are:

- info_box: A mask covering any potential info box in the mikrograph
- material_1_mask: A mask covering all of the first material
- material_1_texture_image: A rectangular mask covering the desired texture example for the first phase
- material_2_texture_image: A rectangular mask covering the desired texture example for the second phase

After these masks have been created the mask_postprocessing can be used to generate a new processed_testing_data sample.
