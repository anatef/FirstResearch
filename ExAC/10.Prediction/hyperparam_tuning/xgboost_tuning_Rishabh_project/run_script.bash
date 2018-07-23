#!/bin/bash

jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1  --execute classification_generic-CV-XGB-Hyperparameter-Tuning-Ran_by_script.ipynb --output param_tuning_output/classification_generic-CV-XGB-Hyperparameter-Tuning-Ran_by_script-$(date '+%d-%m-%Y_%H:%M:%S').ipynb
