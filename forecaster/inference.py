"""Makes a new set of forecasts for price and greenness for all regions

Runs inference on the ML models, returning forecasts. 
"""

import json
import pandas as pd
import numpy as np
import pathlib
import os
np.random.seed(2)

import xgboost
# from fastai.tabular.all import *
import fastai.tabular.all as fastai

# hack workaround for error from pathlib from moving between windows for model training and linux for inference.
# error: cannot instantiate 'WindowsPath' on your system, errorType: NotImplementedError
# from https://github.com/fastai/fastai/issues/1482
# seems it's because some path object is saved in the model file with export()
if fastai.platform.system() == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Constants
MODELS_FOLDER = pathlib.Path('models')
# list of forecasts to make, eg 'VIC1_Price' where forecast is a list of predictions at FORECAST_TIMES (ie up to a week out)
FORECAST_TIMES = list(range(2,24,2)) + list(range(24,168+1,4))  # every 2hrs for 24hrs, then every 4hrs to 168 (week)
ENSEMBLE_RATIO = 0.6  # nn * ENSEMBLE_RATIO  +  xgb (1-ENSEMBLE_RATIO) == prediction

def load_fastai_nn_model(model_id):
    return fastai.load_learner(MODELS_FOLDER / f"{model_id}_nn.pkl")

def load_xgb_model(model_id):
    xgb = xgboost.XGBRegressor()
    xgb.load_model(MODELS_FOLDER / f"{model_id}_xgb.txt")
    xgb.predictor = 'cpu_predictor'  # would be gpu_predictor otherwise, don't know if gpu is available on lambda
    return xgb

def predict_xgb(model_id, data):
    model = load_xgb_model(model_id)
    features = pd.DataFrame([data])
    return model.predict(features)[0]

def predict_fastai_nn(model_id, data):
    # model that predicts a single valu
    model = load_fastai_nn_model(model_id)
    features = pd.Series(data)
    return model.predict(features)[2].item()

def predict_multiple_fastai_nn(model_id, data):
    # model that predicts multiple values
    model = load_fastai_nn_model(model_id)
    features = pd.Series(data)
    return model.predict(features)[2].tolist()




def make_forecast(fc_name, features):
    """Inference for XGB and NN for a single region, either greenness or price. 
    Returns a single flat dict of predictions, with items for each feature predicted by each model at each FORECAST_TIME
    Includes results from both types of model and the ensembled prediction
    """
    print(f"Forecasting {fc_name}")

    predictions = {}
    for fc_time in FORECAST_TIMES:
        model_id = f"{fc_name}_Tp{fc_time}"
        
        # get feature names for this particular model_id - by loading the fastai model and asking it!
        temp_learner = load_fastai_nn_model(model_id)
        data_for_this_model = {key: features[key] for key in temp_learner.dls.cont_names}
    
        if 'Price' in fc_name:
            predictions[f"{model_id}_xgb"] = predict_xgb(model_id, data_for_this_model)
            predictions[f"{model_id}_nn"] = predict_fastai_nn(model_id, data_for_this_model)
            predictions[f"{model_id}_pred"] = (predictions[f"{model_id}_xgb"] * (1-ENSEMBLE_RATIO) + 
                                               predictions[f"{model_id}_nn"] * ENSEMBLE_RATIO)
        else:
            # Greenness - no xgb, multiple outputs
            y_names = temp_learner.dls.y_names
            predictions = predictions | {f"{y_name}": pred for y_name, pred in zip(y_names, predict_multiple_fastai_nn(model_id, data_for_this_model))}
    return predictions
