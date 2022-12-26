"""Makes a new set of forecasts for price and greenness for all regions. 

Gathers up-to-date features from get_features, runs them through the models saved in the container image alongside this and
calculates the forecasts. These are saved in a .json file to S3.
"""

import json
import pandas as pd
import numpy as np
import pathlib
import boto3
from pathlib import Path
import os
import io
import pickle
import requests
import logging 
import importlib

import aemo # only doing this to monkey patch the cache directory, see set_temp_directory()
import get_features

if importlib.util.find_spec('xgboost') is not None: 
    import inference
else:
    logging.info("Running with mock inference because xgboost library not present. Results will be random; won't be saved to S3.")
    import mock_inference as inference  # use for testing to skip inference

logging.basicConfig(level=logging.INFO)

# Constants
REGIONIDS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
# list of forecasts to make, eg 'VIC1_Price' where forecast is a list of predictions at FORECAST_TIMES (ie up to a week out)
FORECASTS_TO_MAKE = [f"{region}_{price_or_greenness}" for region in REGIONIDS for price_or_greenness in ['Price', 'Greenness']]


def main(event=None, context=None):
    """main() is the top level function for the forecaster. It's called by lambda (triggered periodically)
    It uploads the calculated forecasts to the live s3 website bucket
    Also uploads forecasts and calculated features to the history bucket
    Return value is ignored."""

    set_temp_directory()

    # base_time is the (rounded) time when this forecast was made... forecasts actually made a little past the hour but rounded down.
    base_time = pd.Timestamp.now(tz='Australia/Brisbane').round('H').tz_localize(tz=None)

    # collect all the current data 
    features, weather_data, recent_prices, recent_gen_by_fuel, recent_greenness = get_features.get_features()
    print(f"got {len(features)} features")

    # reformat data for json
    weather = format_weather(base_time, weather_data)
    past_data = get_recent_data(base_time, features, recent_prices, recent_greenness)
    past_gen_by_fuel = format_gen_by_fuel(base_time, recent_gen_by_fuel)

    # make forecasts
    forecasts = {}
    for fc in FORECASTS_TO_MAKE:
        forecasts[fc] = inference.make_forecast(fc, features)

    forecast_data, forecast_gen_by_fuel = interpolate_forecasts(base_time, forecasts)

    maximums, minimums = day_max_mins(base_time, past_data, forecast_data)

    output = {
        'baseTimeNem': base_time.isoformat(),
        'baseTimeUtc': nem_time_to_utc_string(base_time),
        'forecastTimestampsUtc': pd.to_datetime(forecast_data.index).map(nem_time_to_utc_string).to_list(),
        'pastTimestampsUtc': pd.to_datetime(past_data.index).map(nem_time_to_utc_string).to_list(),
        'forecasts': {col: forecast_data[col].values.round(2).tolist() for col in forecast_data.columns},
        'past': {col: past_data[col].values.round(2).tolist() for col in past_data.columns},
        'weather': weather,
        'dayMaxs': maximums,
        'dayMins': minimums,
        'recommendations': recommendations(forecast_data),
        'highestEverGreenness': highest_ever_greenness(recent_greenness), 
        'pastGenByFuel': past_gen_by_fuel,
        'forecastGenByFuel': forecast_gen_by_fuel,
    }

    if not hasattr(inference, 'xgboost'):  # test if getting fake data 
        logging.info("forecaster.py Saving to greenforecast.test on S3 because inference is mocked.")
        # set test filename, not production
        bucket = 'greenforecast.test'
    elif 'GREENFORECAST_TEST' in os.environ: 
        logging.info("forecaster.py Saving to greenforecast.test on S3 because we're in a test environment.")
        bucket = 'greenforecast.test'
    else:
        bucket = 'greenforecast.au'

    deploy_forecasts(output, bucket=bucket)

    # save raw data to db if we're in production only
    if bucket == 'greenforecast.au':
        log_features_and_predictions_to_db(base_time, features, forecasts, output)

    return {
        'statusCode': 200,
        'body': 'OK'
    }


def set_temp_directory():
    # if 'RUNNING_LOCALLY' in os.environ:
        # we're running in docker locally, not on AWS.
    if 'AWS_LAMBDA_FUNCTION_VERSION' in os.environ:
        # we're running in from the container (could be locally or on AWS)
        # because it's a read-only filesystem, make sure download caches are in /tmp/
        Path('/tmp/forecaster_cache').mkdir(parents=True, exist_ok=True)  # make sure download cache folder exists
        # monkeypatching aemo.py here, but not actually used in this module. 
        aemo.DOWNLOAD_CACHE_FOLDER = Path('/tmp/forecaster_cache')
        # get_features.DOWNLOAD_CACHE_FOLDER = 

def format_weather(base_time, weather_forecasts):
    """ Parse the weather data to send to the app. Short labels for each day, the weather icon, min/max temp. 
    """
    # day_labels is a list of the day of week in short form. eg ['Today', 'Mon', ...]
    day_labels = [f"Last {(base_time + pd.Timedelta(i, 'D')).strftime('%a')}" for i in range(-7, 0)]
    day_labels = day_labels + [(base_time + pd.Timedelta(i, 'D')).strftime('%a') for i in range(0, 8)]
    day_labels[6] = 'Yesterday'
    day_labels[7] = 'Today'

    output = {
        'dayLabels': day_labels,
    }
    # get weather observations for the last 7 days
    weather_past = get_recent_weather(base_time)

    for region in REGIONIDS:
        # output[region] = [day['icon_descriptor'] for day in weather[region]]
        region_data = []
        weather = weather_past[region] + weather_forecasts[region]
        for day_label, day in list(zip(day_labels, weather)):
            region_data.append({
                'dayLabel': day_label,
                'icon': day['icon_descriptor'],
                'max_temp': day['temp_max'],
                'min_temp': day['temp_min'],
            })
        output[region] = region_data
    return output    

def get_recent_weather(base_time):
    """get_recent_weather() gets the past 7 days of weather, from 7 days befoe base_time to the day before base_time inclusive. 
    Returns a dict of regions, each of which is len(7) list of day records (dict).

    TODO - currently blank
    """
    output = {}
    for region in REGIONIDS:
        output[region] = [{'icon_descriptor': '', 'temp_max': '-', 'temp_min': '-' } for _ in range(7)]

    return output

def get_recent_data(base_time, features, recent_prices, recent_greenness):
    """get_recent_data() gets and formats the recent greenness and price for display on the website
    most of the data is already collected by get_features but not prices for the current day.
    For price, using some other AEMO website that has last 24h conveniently available. 
    Returns a dataframe from 7 days before the current (base_time) day to base_time (inclusive) with columns for all greenness and price features (10 total)
    """

    # last 24h of prices, because they're not in daily reports (ie recent_prices) yet
    r = requests.post('https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN', json={"timeScale":["30MIN"]})
    r = json.loads(r.text)
    df = pd.DataFrame(r['5MIN'])
    df.SETTLEMENTDATE = pd.to_datetime(df.SETTLEMENTDATE)

    assert len(df) > 1000, f"Seems to be a problem with the 5min price data from the AEMO app, expected to have 1725 rows but has {len(df)}"

    df = df[df.PERIODTYPE == 'ACTUAL'] # drop predispatch forecasts
    df = df.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='RRP') 

    df.index = df.index + pd.tseries.frequencies.to_offset('55min')  # because SETTLEMENTDATE is end-of-period and resample will floor minutes
    df = df.resample('H').mean()

    # prices has columns: SETTLEMENTDATE NSW1 QLD1 SA1 TAS1 VIC1, 1 row per hour. in NEM timezone.
    prices = pd.concat([recent_prices, df])
    prices = prices.reset_index().drop_duplicates(['SETTLEMENTDATE'], keep='last') 
    prices = prices.set_index('SETTLEMENTDATE')
    prices.columns = [f"{x}_Price" for x in prices.columns]

    if base_time not in prices.index:
        prices.loc[base_time] = {f"{region}_Price": features[f"{region}_Price"] for region in REGIONIDS}

    # add 30 mins to recent_greenness timestamps and then resample to 1hour
    # it's currently in 30min steps, but (as always) the timestamps are end-of-period
    # we're about to resample to 1 hour, so add 30 mins to make sure each one is in the right bucket
    greenness_hourly = recent_greenness.copy()
    greenness_hourly.index = greenness_hourly.index + pd.tseries.frequencies.to_offset('30min') 
    greenness_hourly = greenness_hourly.resample('1H').mean()

    # combine and interploate to 1 hour (fills in gaps too)
    df = pd.concat([prices, greenness_hourly], axis=1)
    df = df.resample('1H').interpolate(method='linear')

    start = base_time.normalize() - pd.Timedelta(7, 'D') # aligned with past_gen_by_fuel
    df = df[start:base_time + pd.Timedelta(1, 'min')]

    df = df.round(3)
    return df

def format_gen_by_fuel(base_time, recent_gen_by_fuel):
    """ format_gen_by_fuel() takes the data calculated by get_greenness (greenness inputs) and formats it for the 
    website to present pretty stacked area charts of historical data.
    """
    df = recent_gen_by_fuel.copy()
    start = base_time.normalize() - pd.Timedelta(7, 'D') # aligned with past_data

    # add 30 mins to recent_greenness timestamps and then resample to 1hour
    # it's currently in 30min steps, but (as always) the timestamps are end-of-period
    # we're about to resample to 1 hour, so add 30 mins to make sure each one is in the right bucket
    df.index = df.index + pd.tseries.frequencies.to_offset('30min') 
    df = df[start:].resample('1H').mean()

    return convert_gen_by_fuel_to_percentage(df)

def convert_gen_by_fuel_to_percentage(df):

    output = {}
    for region in REGIONIDS:

        all_gen_names = [f'{region}_GEN_{fuel}' for fuel in ['Coal', 'Gas', 'Hydro', 'Solar', 'Wind', 'Rooftop']] + [f'{region}_IC_Green_In', f'{region}_IC_Fossil_In']
        all_gen = df[all_gen_names].sum(axis=1)

        percentage = df[all_gen_names].div(all_gen, axis=0) * 100
        percentage = percentage.round(1)
        percentage.columns = [s.replace(f"{region}_", "").replace(f"GEN_", "").replace(f"IC_", "") for s in percentage.columns]

        output[region] = { col: percentage[col].values.round(2).tolist() for col in percentage.columns }

    output['timestampsUtc'] = percentage.index.map(nem_time_to_utc_string).to_list()

    return output

def interpolate_forecasts(base_time, forecasts): 
    """ interpolate_forecasts() puts the flat dict of feature predictions into a dataframe, adding an absolute time index on the way
    returns tuple of 
    - dataframe of price/greenness forecasts as a dict of region:forecasts
    - gen by fuel, as a nested dict, formatted for output in the json
    """

    # turn each forecast into a list, taking only the emsembled prediction ('..._pred') not 
    # the raw xgb/nn for each time. 
    # for the fuel features, extract them into their own columns. 
    columns = {}
    for fc_name, fc in forecasts.items():
        if 'Price' in fc_name:
            columns[fc_name] = [val for key, val in fc.items() if '_pred' in key]
        else:
            # Greenness features (gen by fuel)
            fuel_names = [x.split('_Tp84')[0] for x in fc if '_Tp84' in x]
            for fuel in fuel_names:
                columns[fuel] = [val for key, val in fc.items() if fuel in key]

    df = pd.DataFrame(columns, index=inference.FORECAST_TIMES)

    # change to datetime index
    df.index = [base_time + pd.Timedelta(hours, 'H') for hours in df.index]
    
    # record max/min before interpolation, for an error check
    dfmax, dfmin = df.max().max(), df.min().min()

    # interpolate everything
    df = df.resample('1H').interpolate(method='cubic')
    df = df.round(3)

    # had an error interpolating with method='spline', curve went crazy high/low. Throw if that happens. 
    if df.max().max() > dfmax + 1000 or df.min().min() < dfmin - 1000:
        raise RuntimeError('interpolate() has gone nuts')

    # calculate predicted greenness from the predicted gen by fuel (currently, df has price and fuel features only)
    for region in REGIONIDS:
        renewable = [f'{region}_GEN_{fuel}' for fuel in ['Hydro', 'Rooftop', 'Solar', 'Wind']] + [f'{region}_IC_Green_In']
        all_gen = [f'{region}_GEN_{fuel}' for fuel in ['Coal', 'Gas', 'Hydro', 'Rooftop', 'Solar', 'Wind']] + [f'{region}_IC_Fossil_In', f'{region}_IC_Green_In']

        df[all_gen] = df[all_gen].clip(lower=0)

        ####################
        ### HACK to fix bug from model trained on dataset8
        ### y_names are wrong in the model files beacuse of bug in training notebook - the features are mislabelled
        ### The labels should be: ['Coal', 'Gas', 'Hydro', 'Solar', 'Wind', 'Rooftop']] + [f'{region}_IC_Green_In', f'{region}_IC_Fossil_In']
        ### ... but the labels actually are sorted alphabetically, so solar/wind/rooftop are rotated one and green_in/fossil_in are swapped.
        ### This hack reverses these.
        ### the bug is now fixed in the trainer, so remove when the next model is trained. 
        ### in future, it will actually be correct for the labels to be sorted alphabetically, ie data will be shifted but labels won't be changed.
        df.rename(columns={
            f'{region}_GEN_Rooftop': f'{region}_GEN_Solar',
            f'{region}_GEN_Solar': f'{region}_GEN_Wind',
            f'{region}_GEN_Wind': f'{region}_GEN_Rooftop',
            f'{region}_IC_Green_In': f'{region}_IC_Fossil_In',
            f'{region}_IC_Fossil_In': f'{region}_IC_Green_In',
        }, inplace=True)
        ### end hack. 
        #########

        # calculate greenness from fuel features
        df[f'{region}_Greenness'] = df[renewable].sum(axis=1) / df[all_gen].sum(axis=1) * 100

    # print(df[[x for x in df.columns if 'NSW1' in x]])
    # print(forecasts['NSW1_Greenness'])

    price_and_greenness_columns = [col for col in df.columns if 'Price' in col or 'Greenness' in col]
    gen_by_fuel_percent = convert_gen_by_fuel_to_percentage(df[[col for col in df.columns if col not in price_and_greenness_columns]])
    return df[price_and_greenness_columns], gen_by_fuel_percent

def day_max_mins(base_time, past_data, forecasts):
    """day_max_mins() calculates the max and min for each forecast for each day (in nem timezone) and also reports the time (in utc) it happens 
    NOTE: this uses NEM time not local time. So if the max is either side of mightnight then may have wrong result. Accepting this for now. TODO.
    """
    df = pd.concat([past_data, forecasts])
    
    days = [(base_time + pd.Timedelta(x, 'D')).date().isoformat() for x in range(-7, 8)]
    
    maximums = {key: [] for key in df.columns}
    minimums = {key: [] for key in df.columns}
    for day in days:
        max_times = df.loc[day].idxmax()
        maxs = df.loc[day].max()
        for col, m in maxs.items():
            maximums[col].append({
                'utc': max_times[col].tz_localize(tz='Australia/Brisbane').tz_convert('UTC').isoformat(), 
                'max': m
            })
        min_times = df.loc[day].idxmin()
        mins = df.loc[day].min()
        for col, m in mins.items():
            minimums[col].append({
                'utc': min_times[col].tz_localize(tz='Australia/Brisbane').tz_convert('UTC').isoformat(), 
                'min': m
            })
    return maximums, minimums

def recommendations(forecast_data):
    """Returns specific recommendations for when to charge evs or when to reduce usage, in the next 24 hrs. Dictionary for export in the .json
    Currently hardcoded, TODO: automate
    """
    ev = {}
    for region in REGIONIDS:
        ev[region] = 'Overnight' if region == 'TAS1' else 'Daytime'

    reduce_usage = {}
    for region in REGIONIDS:
        reduce_usage[region] = 'Daytime' if region == 'TAS1' else 'Overnight'

    return {
        'ev': ev,
        'reduceUsage': reduce_usage,
    }

def highest_ever_greenness(recent_greenness):
    """Get and return the record max greenness. 

    Reads stored state from S3, checks if new data has anything higher, then writes back to S3 and returns the dict result. Example format: 
    {'NSW1': {'value': 71, 'utc': '2022-11-20T01:00:00+00:00'},
     'QLD1': {'value': 70, 'utc': '2022-11-20T01:00:00+00:00'},
     'SA1': {'value': 100, 'utc': '2022-11-20T01:00:00+00:00'},
     'TAS1': {'value': 100, 'utc': '2022-11-20T01:00:00+00:00'},
     'VIC1': {'value': 73, 'utc': '2022-11-20T01:00:00+00:00'}}
    """

    # grab current highest-ever from s3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='greenforecast.forecaster', Key='highest_ever_greenness')
    highest = json.load(response['Body'])

    # compare against the new data, update if found a new winner
    highest_recent = recent_greenness.max()
    highest_recent_idx = recent_greenness.idxmax()
    for region in REGIONIDS:
        if highest_recent[f'{region}_Greenness'] > highest[region]['value']+0.5:
            print(f"New Highest Greenness for {region}: {highest_recent[f'{region}_Greenness']}")
            highest[region]['value'] = int(np.round(highest_recent[f'{region}_Greenness']))
            highest[region]['utc'] = nem_time_to_utc_string(highest_recent_idx[f'{region}_Greenness'])

    # write back to s3
    bucket = boto3.resource('s3').Bucket('greenforecast.forecaster')
    bucket.put_object(Key='highest_ever_greenness', Body=json.dumps(highest))

    return highest


def deploy_forecasts(forecasts, bucket='greenforecast.au', filename='latest_forecasts.json'):
    """ deploy_forecasts() exports all the forecasts just made to S3 """
    
    s3 = boto3.resource('s3')

    website = s3.Bucket(bucket)
    website.put_object(Key=filename, Body=json.dumps(forecasts, cls=NpEncoder))

def log_features_and_predictions_to_db(base_time, features, forecasts, interpolated):
    """log_features_and_predictions_to_db() takes the features pulled from aemo and the 
    results of the predictions and saves them to dynamodb for reference later. 
    features = {'VIC1_Price': 100, ...}
    a forecast is a dict of predictions, forecasts is a dict of forecasts"""

    # # join each forecast together into a single dict
    # # predictions = {'VIC1_Price_Tp2_pred': 1000, ... 'VIC1_Greenness_Tp168_pred': 1000, ... 'NSW1_Price_Tp84_pred': 100, ...}
    # predictions = {}
    # for preds in forecasts.values():
    #     predictions = predictions | preds
        
    # # join input features and predictions
    # all_features = dict(sorted(list(features.items()) + list(predictions.items())))
    
    # print(f"TODO: write to database {len(all_features)} features, which includes {len(features)} features.")

    # write all_features to database. 
    # key is current time we're forecasting from,
    # value is all_features

    body = {
        'features': features,
        'predictions': forecasts,
        'forecasts': interpolated
    }

    s3 = boto3.resource('s3')
    history = s3.Bucket('greenforecast.history')
    history.put_object(Key=base_time.isoformat(), Body=json.dumps(body, cls=NpEncoder))

def get_test_features():
    with open('test_features.json') as f:
        features = json.load(f)
    return features

def nem_time_to_utc_string(nem_time):
    """ returns isoformat string in utc """
    return nem_time.tz_localize(tz='Australia/Brisbane').tz_convert('UTC').isoformat()

class NpEncoder(json.JSONEncoder):
    """ prevents errors when json.dump-ing numpy objects 
    from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



if __name__ == '__main__':
    # This code is not hit by lambda. Local dev only. 
    # Lambda runs main() directly.
    print("")

    main()


    ########

    # # base_time is the (rounded) time when this forecast was made... will actually be a little past the hour. 
    # base_time = pd.Timestamp.now(tz='Australia/Brisbane').round('H').tz_localize(tz=None)

    # # collect all the current data 
    # features, weather_data, recent_prices, recent_greenness = get_features.get_features()
    # print(f"got {len(features)} features")

    # print(format_weather(base_time, weather_data))
