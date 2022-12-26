"""Mocks inference.py for dev machine that doesn't have ML libraries installed. returns random values in correct format.
"""

import json
import pandas as pd
import numpy as np
import pathlib
import random

# Constants
# list of forecasts to make, eg 'VIC1_Price' where forecast is a list of predictions at FORECAST_TIMES (ie up to a week out)
FORECAST_TIMES = list(range(2,24,2)) + list(range(24,168+1,4))  # every 2hrs for 24hrs, then every 4hrs to 168 (week)
ENSEMBLE_RATIO = 0.6  # nn * ENSEMBLE_RATIO  +  xgb (1-ENSEMBLE_RATIO) == prediction

MOCK_INFERENCE = True  # presence of this indicates that we're giving fake data

def make_forecast(fc_name, features):
    """Inference for XGB and NN for a single region, either greenness or price. 
    Returns a list of predictions, one for each FORECAST_TIME. 
    Includes results from both types of model and the ensembled prediction
    """
    print(f"Forecasting {fc_name}")

    predictions = {}
    for fc_time in FORECAST_TIMES:
        model_id = f"{fc_name}_Tp{fc_time}"
        
        if 'Price' in fc_name:
            predictions[f"{model_id}_xgb"] = random.uniform(0, 100)
            predictions[f"{model_id}_nn"] = random.uniform(0, 100)
            predictions[f"{model_id}_pred"] = (predictions[f"{model_id}_xgb"] * (1-ENSEMBLE_RATIO) + 
                                               predictions[f"{model_id}_nn"] * ENSEMBLE_RATIO)
        else:
            # Greenness features (gen by fuel)
            region = fc_name.split('_')[0]
            y_names = [f'{region}_GEN_Coal_Tp{fc_time}', f'{region}_GEN_Gas_Tp{fc_time}', f'{region}_GEN_Hydro_Tp{fc_time}', f'{region}_GEN_Rooftop_Tp{fc_time}', 
                       f'{region}_GEN_Solar_Tp{fc_time}', f'{region}_GEN_Wind_Tp{fc_time}', f'{region}_IC_Fossil_In_Tp{fc_time}', f'{region}_IC_Green_In_Tp{fc_time}']
            predictions = predictions | {y_name: random.uniform(0, 1000) for y_name in y_names}

    return predictions
