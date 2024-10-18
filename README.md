# VolCon
This repository contains a few-shot volume fraction prediction model (VolCon). The repository deals with training and evaluating this model.

## Setting up the Repository
After Cloning the repository, the dependency an the packages of the repository (training, model and utils) have to be installed the installation is done in the following steps.
```shell
pip install poetry
```
Then install all dependencies, including development and deployment using the following command:
```shell
poetry install
```
If you only want to install the application dependencies (not the development dependencies), the following command will also work:
```shell
pip install .
```

## Structure of the Repository
- The ./model folder contains the model definition of the volcon model
- The ./testing folder contains the data and pipelines used to evaluation / benchmark the trained volcon models
- The ./training folder contains the  pipeline to train a volcon model from scratch only using sythetic data
- The ./utils folder contains utily functions used in the other segments of the repository
