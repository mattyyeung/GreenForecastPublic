""" Gather, calculate and return all the features required to make a prediction, using most recent available data. 

Visits aemo, BoM and uses other data. 
Returns a dict of features, for all states. test_features.json contains an example set. 
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date
import json #temp
import os
import logging
import math

from aemo import get_duids, get_latest_file_from_folder, yield_files_from_folder_by_index, get_subtable_from_file
import get_greenness

REGIONIDS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
DATA_FOLDER = Path('data')
DOWNLOAD_CACHE_FOLDER = Path('/tmp/forecaster_cache')

def get_features():
    """top-level function for this module: collects and returns a full set of features from 
    the latest live aemo/whoever data, ready to feed into forecasting models.
    Also returns full weather forecasts gathered along the way. """

    features = {}

    duids = get_duids()

    Path(DOWNLOAD_CACHE_FOLDER).mkdir(parents=True, exist_ok=True)  # make sure download cache folder exists;
    
    print("\nget_date_features():")
    features = features | get_date_features()
    print("\nget_weather_features():")
    weather_features, weather = get_weather_features()
    features = features | weather_features

    print("\nget_price_and_gen():")
    features = features | get_price_and_gen()
    print("\nget_price_lags():")
    price_features, recent_prices = get_price_lags()
    features = features | price_features
    print("\nget_predispatch_prices():")
    features = features | get_predispatch_prices()
    print("\nget_price_gen_forecasts():")
    features = features | get_price_gen_forecasts()
    print("\nget_availability_by_fuel(duids):")
    features = features | get_availability_by_fuel(duids)
    print("\nget_generation_by_fuel(duids):")
    features = features | get_generation_by_fuel(duids)
    print("\nget_rooftop_pv():")
    features = features | get_rooftop_pv()

    print("\nget_greenness.get_greenness(duids):")
    greenness_features, gen_by_fuel_month, greenness_month = get_greenness.get_greenness(duids)
    features = features | greenness_features

    # check for NaNs
    for val in features.values():
        if math.isnan(val): 
            logging.warning(features)
            raise Exception("Error, there's a NaN in features. All features were just printed, see prev line.")

    return features, weather, recent_prices, gen_by_fuel_month, greenness_month

def get_date_features():
    """Get date features: 'day', 'hour', 'hours_since_2010', 'is_weekend', 'month', 'quarter', 'weekday', 'year'
   Also, get features for holidays and workdays
    """
    output = {}
    now = pd.Timestamp.now(tz='Australia/Brisbane').to_pydatetime()
    output['year'] = now.year - 2010
    output['month'] = now.month
    output['day'] = now.day
    output['hour'] = now.hour
    output['weekday'] = now.weekday()
    output['quarter'] = ((now.month-1) // 3)+1 + (now.year-2010) * 4
    output['is_weekend'] = now.weekday() >= 5

    # hours_since_2010: note, not actually hours, it's timesteps, and they are now 5 min timesteps. 
    # also it's rounded to 3 sig figs as a float... wonder if that's enough...
    # 9 Jan 2010 happens to equal 2020
    START_DATE = datetime(2010, 1, 2)

    def datetime_to_x_since_2010(dt): 
        delta = dt.replace(tzinfo=None) - START_DATE
        x = delta.total_seconds() / 60 / 5
        return int(float('%.3g' % x))
    # tests from dataset6
    assert datetime_to_x_since_2010(datetime(2010, 1, 9)) == 2020
    assert datetime_to_x_since_2010(datetime(2022, 6, 20, 10)) == 1310000
    assert datetime_to_x_since_2010(datetime(2010, 1, 15, 22, 34, 0)) == 4010
    assert datetime_to_x_since_2010(datetime(2010, 1, 15, 22, 35, 0)) == 4020

    output['hours_since_2010'] = datetime_to_x_since_2010(now)
    
    holidays = pd.read_csv(DATA_FOLDER / 'australian_public_holidays_scraped_2010-2024.csv', parse_dates=['Date'], index_col=0)
    holidays = ~holidays.pivot(index='Date', columns='REGIONID').isna()
    holidays.columns = [f"{x[1]}_is_holiday" for x in holidays.columns]

    def get_is_holiday(date, region):
        date_str = date.isoformat()
        if date_str in holidays.index:
            region_is_holiday = holidays.loc[date_str, f'{region}_is_holiday']
        else:
            region_is_holiday = False
        return region_is_holiday

    assert get_is_holiday(date(2010, 1, 26), 'SA1') == True
    assert get_is_holiday(date(2010, 1, 30), 'QLD1') == False
    assert get_is_holiday(date(2024, 11, 4), 'NSW1') == False
    assert get_is_holiday(date(2024, 11, 5), 'VIC1') == True

    # NEM time = AEST = Brisbane, because they don't have daylight savings time
    today = pd.Timestamp.now(tz='Australia/Brisbane').normalize().to_pydatetime().date()
    for region in REGIONIDS:
        output[f'{region}_is_holiday'] = get_is_holiday(today, region)
        output[f'{region}_is_workday'] = not (output['is_weekend'] or output[f'{region}_is_holiday'])

    next_7_days = [today + timedelta(days=x) for x in range(1, 7+1)]
    is_weekend_forecast = [day.weekday() >= 5 for day in next_7_days]
    for region in REGIONIDS:
        is_holiday_forecast = [get_is_holiday(day, region) for day in next_7_days]
        is_workday_forecast = [not(weekend or holiday) for weekend, holiday in list(zip(is_weekend_forecast, is_holiday_forecast))]
        is_workday_forecast
        for i, is_workday in enumerate(is_workday_forecast):
            output[f'{region}_is_workday_Tp{24 * (i+1)}'] = is_workday

    return output

def get_weather_features():
    """get max temp forecasts from the BoM

    returns dict of features for max temp
    also returns all forecast data for processing elsewhere. all_forecasts

    Data comes from the api used in https://weather.bom.gov.au/location/r1r0fsn-melbourne and similar pages
    eg:
    https://api.weather.bom.gov.au/v1/locations/r7hgdp/forecasts/daily
    Dodgy: often short one day... sometimes short two! See below
    """

    """TODO: use the xml on the bom website. Below is some test code for an example. 
    import urllib
    import xml.dom.minidom

    # URL from http://www.bom.gov.au/catalogue/anon-ftp.shtml search for 'city forecast' 
    # we want "IDS10034 -   City Forecast - Adelaide (SA)" 
    with urllib.request.urlopen('ftp://ftp.bom.gov.au/anon/gen/fwo/IDV10450.xml') as f:
        # the_page = f.read(1000)
        
        # Open XML document using minidom parser
        DOMTree = xml.dom.minidom.parse(f)
        collection = DOMTree.documentElement
    collection
    region = 'VIC1'

    aacs = {
        'NSW1': 'NSW_PT131',
        'VIC1': 'VIC_PT042',
    }

    def getElementByAttribute(collection, tag_name, attribute, value):
        elements = collection.getElementsByTagName(tag_name)
        for el in elements:
            print(el.getAttribute(attribute), " ", end="")
            if el.getAttribute(attribute) == value:
                return el
        return None

    city = getElementByAttribute(collection, 'area', 'aac', aacs[region])
    text = city.getElementsByTagName('element')
    forecast_icon_codes = [el.firstChild.data for el in text if el.getAttribute('type') == 'forecast_icon_code']
    forecast_icon_codes
    """

    bom_codes = {
        'NSW1': 'r3gx2f',
        'VIC1': 'r1r0fs',
        'TAS1': 'r22u09',
        'QLD1': 'r7hgdp',
        'SA1': 'r1f93c',
    }
    all_forecasts = {}
    output = {}
    for region in REGIONIDS:
    # for region in ['SA1']:

        # get current temperature
        r = requests.get(f'https://api.weather.bom.gov.au/v1/locations/{bom_codes[region]}/observations')
        observation = r.json()
        output[f'{region}_W_temperature'] = observation['data']['temp']
        max_temp_today = observation['data']['max_temp']['value']
    
        # get forecast temperatures
        r = requests.get(f'https://api.weather.bom.gov.au/v1/locations/{bom_codes[region]}/forecasts/daily')
        weather = r.json()

        # extract date & max_temp from the BoM data.  Round to nearest day to avoid having to deal with DST or each city's particular timezone.
        weather_by_date = {pd.Timestamp(x['date']).tz_convert('Australia/Brisbane').round('D').tz_localize(tz=None): x 
                        for x in weather['data']}

        # need to match those dates against the desired dates, which start from 'today' (regardless of current time) and then 7 more days after that.
        today = (pd.Timestamp.utcnow()
                 .tz_convert('Australia/Brisbane')  # Use Brisbane tz beacuse always AEST (no DST)
                 .floor('D')
                 .tz_localize(tz=None))  # remove timezone
        # days is a list of the days we want forecasts for. That is, today and the following 7 days.
        days = [today + pd.Timedelta(x, 'D') for x in range(8)]

        # print(f"{region} {days}")

        if today not in weather_by_date: weather_by_date[today] = None
        if days[-2] not in weather_by_date: weather_by_date[days[-2]] = None  # if there are only 6 days total, not 8
        if days[-1] not in weather_by_date: weather_by_date[days[-1]] = None  # if there are only 7 
        forecasts = [weather_by_date[day] for day in days]
        # dodgy hack if there's one or two days short: copy from previous day
        # dataset7 has current day's max temp (depending on current time, this could be future or 
        # past) and requests the next 7 days ... but sometimes (eg early in the morning), there aren't this many days forecast by bom, it returns 'None' in last day.
        # So just copy the prev day's prediction. 
        if forecasts[-2] is None: forecasts[-2] = forecasts[-3].copy()
        if forecasts[-1] is None: forecasts[-1] = forecasts[-2].copy()
        # Sometimes (1am mon) there IS a response for the 7th day but vaues are None. If htis happens just copy prev day.
        if None in forecasts[-2].values(): forecasts[-2] = forecasts[-3].copy()
        if None in forecasts[-1].values(): forecasts[-1] = forecasts[-2].copy()
        #  if today is missing (TODO: seems to happen at midnight-1am AEDST during daylight savings) then replace it with observed data from today
        if forecasts[0] is None: 
            # import IPython; IPython.embed() # debug
            forecasts[0] = forecasts[1].copy() 
            forecasts[0]['temp_max'] = max_temp_today
            print(f'===Oops: In get_weather_forecasts(), "today" is missing. Today: {today}. Continuing anyway.')
        # forecasts[0] = max(forecasts[0], max_temp_today)  # don't do this because observations may be for yesterday (they are at 1am)
        
        assert None not in forecasts, f"something has gone wrong with weather forecast data for {region}. Forecasts are {forecasts}"
        assert len(forecasts) == 8, f"there should be 8 forecast {region} temperatures but there are {len(forecasts)}"

        temp_names = ['RG_W_day_max_temperature', 'RG_W_day_max_temperature_Tp24', 'RG_W_day_max_temperature_Tp48', 'RG_W_day_max_temperature_Tp72', 'RG_W_day_max_temperature_Tp96', 'RG_W_day_max_temperature_Tp120', 'RG_W_day_max_temperature_Tp144', 'RG_W_day_max_temperature_Tp168']
        temp_names = [x.replace('RG', region) for x in temp_names]
        
        for name, forecast in list(zip(temp_names, forecasts)):
            output[name] = forecast['temp_max']

        all_forecasts[region] = forecasts

    return output, all_forecasts

def get_price_and_gen():
    """Get Price & Generation features by region for the current time

    Columns: Price, IC_NET, GENERATION, IC_Import_Limit, IC_Export_Limit
    Downloads various sub-tables of "Dispatch", which are equivalent of `DISPATCHREGIONSUM`, 
    `DISPATCHPRICE` and `DISPATCHINTERCONNECTORRES` in the dataset generator.
    """
    output = {}
    # price features
    filename = get_latest_file_from_folder('DispatchIS_Reports')
    price_table = pd.read_csv(get_subtable_from_file(filename, 'DISPATCH,PRICE,'), 
                              usecols=['REGIONID', 'RRP'])
    price_table = price_table.drop_duplicates(['REGIONID'], keep='last')
    price_table = price_table.set_index('REGIONID')

    for region in REGIONIDS:
        output[f'{region}_Price'] = price_table.at[region, 'RRP']

    # regionsum features aka 'DISPATCHREGIONSUM' aka 'Generation For Each Region'
    regionsum_table = pd.read_csv(get_subtable_from_file(filename, 'DISPATCH,REGIONSUM,'),
                                  usecols=['REGIONID', 'AVAILABLEGENERATION', 'TOTALDEMAND', 'DISPATCHABLEGENERATION', 'NETINTERCHANGE'])
    # drop duplicates because there can be multiple different runs ('runid')
    regionsum_table = regionsum_table.drop_duplicates(['REGIONID'], keep='last')
    regionsum_table = regionsum_table.set_index('REGIONID')

    # IC_NET is defined to be NETINTERCHANGE * -1. IC_NET: import is +ve generation.
    regionsum_table['NETINTERCHANGE'] = regionsum_table['NETINTERCHANGE'] * -1
    regionsum_table = regionsum_table.rename(columns={'NETINTERCHANGE': 'IC_NET', 'DISPATCHABLEGENERATION': 'GENERATION' })
    for region in REGIONIDS:
        for col in regionsum_table.columns:
            output[f'{region}_{col}'] = regionsum_table.at[region, col]

    # Interconnector import / export limits
    ic_table = pd.read_csv(get_subtable_from_file(filename, 'DISPATCH,INTERCONNECTORRES,'), 
                           usecols=['INTERCONNECTORID', 'EXPORTLIMIT', 'IMPORTLIMIT'])
    # drop duplicates because there can be multiple different runs ('runid')
    ic_table = ic_table.drop_duplicates(['INTERCONNECTORID'], keep='last')
    ic_table = ic_table.set_index('INTERCONNECTORID')

    # manually convert each interconnector data to region summaries
    output['VIC1_IC_Import_Limit'] = (- ic_table.at['T-V-MNSP1', 'EXPORTLIMIT'] + ic_table.at['V-S-MNSP1', 'IMPORTLIMIT'] + ic_table.at['V-SA', 'IMPORTLIMIT'] + ic_table.at['VIC1-NSW1', 'IMPORTLIMIT']) * -1
    output['VIC1_IC_Export_Limit'] = (- ic_table.at['T-V-MNSP1', 'IMPORTLIMIT'] + ic_table.at['V-S-MNSP1', 'EXPORTLIMIT'] + ic_table.at['V-SA', 'EXPORTLIMIT'] + ic_table.at['VIC1-NSW1', 'EXPORTLIMIT']) * -1
    output['TAS1_IC_Import_Limit'] = ic_table.at['T-V-MNSP1', 'IMPORTLIMIT'] * -1
    output['TAS1_IC_Export_Limit'] = ic_table.at['T-V-MNSP1', 'EXPORTLIMIT'] * -1
    output['SA1_IC_Import_Limit']  = (- ic_table.at['V-S-MNSP1', 'EXPORTLIMIT'] - ic_table.at['V-SA', 'EXPORTLIMIT']) * -1
    output['SA1_IC_Export_Limit']  = (- ic_table.at['V-S-MNSP1', 'IMPORTLIMIT'] - ic_table.at['V-SA', 'IMPORTLIMIT']) * -1
    output['NSW1_IC_Import_Limit'] = (- ic_table.at['VIC1-NSW1', 'EXPORTLIMIT'] + ic_table.at['NSW1-QLD1', 'IMPORTLIMIT'] + ic_table.at['N-Q-MNSP1', 'IMPORTLIMIT']) * -1
    output['NSW1_IC_Export_Limit'] = (- ic_table.at['VIC1-NSW1', 'IMPORTLIMIT'] + ic_table.at['NSW1-QLD1', 'EXPORTLIMIT'] + ic_table.at['N-Q-MNSP1', 'EXPORTLIMIT']) * -1
    output['QLD1_IC_Import_Limit'] = (- ic_table.at['NSW1-QLD1', 'EXPORTLIMIT'] - ic_table.at['N-Q-MNSP1', 'EXPORTLIMIT']) * -1
    output['QLD1_IC_Export_Limit'] = (- ic_table.at['NSW1-QLD1', 'IMPORTLIMIT'] - ic_table.at['N-Q-MNSP1', 'IMPORTLIMIT']) * -1
    return output

def get_price_lags():
    """Get all the price lag features."""
    output = {}

    print('Get lags from the past 24h:')
    # The last 24 hours of lags aren't necessarily in the daily reports, so get them from 5-minute reports.
    # Note: Not bothering to do any averaging across the whole hour, just instantaneous 5-min values for < 24h.
    lags = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    targets = [f'RG_Price_Tm{hours}' for hours in lags]

    filenames = yield_files_from_folder_by_index('TradingIS_Reports', [-1*12*x for x in lags])

    for lag, filename in zip(lags, filenames):
        df = pd.read_csv(get_subtable_from_file(filename, ',TRADING,PRICE,'),
                         usecols=['REGIONID', 'RRP', ]
                        )
        # drop duplicates because there can be multiple different runs ('runid')
        df = df.drop_duplicates(['REGIONID'], keep='last')
        df = df.set_index('REGIONID')

        for region in REGIONIDS:
            output[f"{region}_Price_Tm{lag}"] = df.at[region, 'RRP']
            
    # Get the remaining lags up to 30 days ago from the daily reports
    # 'RG_Price_Tm28', 'RG_Price_Tm32',... 'RG_Price_Tm168',
    # 'RG_Price_Tm30d_25thP', 'RG_Price_Tm30d_Median', 'RG_Price_Tm30d_75thP', 'RG_Price_Tm30d_Mean'
    print('Get last 30 days of daily reports:')
    filenames = yield_files_from_folder_by_index('Daily_Reports', range(-1, -31, -1))
    dfs = []
    for filename in filenames:
        # Get prices, average to hourly, for the last 30 days
        df = pd.read_csv(get_subtable_from_file(filename, ',DREGION,,3'),
                         usecols=['REGIONID', 'RRP', 'SETTLEMENTDATE', ],
                         parse_dates=['SETTLEMENTDATE']
                        )
        # remove duplicates because one day there were multiples in another file for some reason. It happened with PUBLIC_PREDISPATCHIS_202211151030_20221115100251.CSV. 
        df = df.drop_duplicates(['SETTLEMENTDATE', 'REGIONID'], keep='last')

        df = df.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='RRP')
        # change from 5min to 60min frequency to match PV_forecast. Take average across the hour.
        # df = df.resample('H', origin='start').mean()
        # df.index = df.index + pd.tseries.frequencies.to_offset('55min')  # because SETTLEMENTDATE is end-of-period
        # don't care what the current minutes are, average to integer hours. 
        df.index = df.index + pd.tseries.frequencies.to_offset('55min')  # because SETTLEMENTDATE is end-of-period and resample will floor minutes
        df = df.resample('H').mean()

        dfs.append(df)

        # reduce max ephemeral space (in /tmp) used by the lambda function by deleting these big .csvs as we go. 
        # this works because yield_files_from_folder_by_index is a generator. 
        # print(f"Delete {filename}")
        os.remove(filename)

    # concat all
    df = pd.concat(dfs)
    df = df.reset_index().drop_duplicates(['SETTLEMENTDATE'], keep='last') # daily reports overlap each other
    df = df.set_index('SETTLEMENTDATE')

    now = (pd.Timestamp.utcnow()
           .tz_convert('Australia/Brisbane')  # Use Brisbane tz because it's always AEST == NEM time.
           .floor('H')
           .tz_localize(tz=None))  # remove timezone
    # 'RG_Price_Tm28', 'RG_Price_Tm32',... 'RG_Price_Tm168',
    for lag in [28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 168]:
        for region in REGIONIDS:
            output[f"{region}_Price_Tm{lag}"] = df.at[now - pd.Timedelta(lag, 'H'), region]
    
    # 'RG_Price_Tm7d_25thP', 'RG_Price_Tm7d_Median', 'RG_Price_Tm7d_75thP'
    for region in REGIONIDS:
        last_7_days = df[region][df.index > now - pd.Timedelta(7, 'D')]
        output[f'{region}_Price_Tm7d_15thP'] = last_7_days.quantile(0.15)
        output[f'{region}_Price_Tm7d_Median'] = last_7_days.quantile(0.5)
        output[f'{region}_Price_Tm7d_85thP'] = last_7_days.quantile(0.85)

    # 'RG_Price_Tm30d_25thP', 'RG_Price_Tm30d_Median', 'RG_Price_Tm30d_75thP'
    for region in REGIONIDS:
        output[f'{region}_Price_Tm30d_15thP'] = df[region].quantile(0.15)
        output[f'{region}_Price_Tm30d_Median'] = df[region].quantile(0.5)
        output[f'{region}_Price_Tm30d_85thP'] = df[region].quantile(0.85)
    
    return output, df

def get_predispatch_prices():
    """Get predispatch price prediction features
    
    'VIC1_Predis_Price_Tp4', 'VIC1_Predis_Price_Tp8', 'VIC1_Predis_Price_Tp12',  'VIC1_Predis_Price_Tp16',
    'VIC1_Predis_Price_max16to40h', 'VIC1_Predis_Price_min16to40h'
    """
    filename = get_latest_file_from_folder('PredispatchIS_Reports')
    df = pd.read_csv(get_subtable_from_file(filename, 'PREDISPATCH,REGION_PRICES'),
                     usecols=['REGIONID', 'RRP', 'DATETIME'],
                     parse_dates=['DATETIME'])

    # Duplicated rows ... probably because of multiple (types of?) runs for same period. Keep only the last ones. 
    df = df.drop_duplicates(['DATETIME', 'REGIONID'], keep='last')
    # Don't need half-hourly, hourly is fine, drop the other rows
    df = df[df['DATETIME'].dt.minute == 0]

    # do the pivot
    now = df.iloc[0].DATETIME
    df['Forecast_Distance'] = df['DATETIME'] - now
    df = df.pivot(index='Forecast_Distance', 
                  columns='REGIONID',
                  values='RRP'
                 ).sort_index(axis=1, level='REGIONID')
    df.index = df.index.total_seconds() // 3600

    logging.info(f"get_predispatch_prices() df.shape={df.shape} (because sometimes 1pm doesn't work?) ")
    # we're assuming that 16 entries are present...  might not be true right before a predispatch run? 
    last_index = 16
    if len(df) <= last_index:
        logging.warning(f"problem: there isn't enough predispatch. Its shape is: {df.shape}. Just using the last available one instead of 16.")
        last_index = len(df) - 1

    output = {}
    for region in REGIONIDS:
        for lag in [4, 8, 12, 16]:
            output[f'{region}_Predis_Price_Tp{lag}'] = df[region].loc[min(lag, last_index)]
        output[f'{region}_Predis_Price_max16to40h'] = df[region].loc[last_index:44].max()
        output[f'{region}_Predis_Price_min16to40h'] = df[region].loc[last_index:44].min()
    return output

def get_price_gen_forecasts():
    """Get features for forecasts of Price & Gen by Region.

    Combination pre-dispatch (for the next day-or-so) and STPASA (which starts at 4.30am on day after next trading day)
    Outputs dict of features:
        'RG_AVAILABLEGENERATION_Tp6', 'RG_AVAILABLEGENERATION_Tp12'...'RG_AVAILABLEGENERATION_Tp168',
        'RG_TOTALDEMAND_Tp6', 'RG_TOTALDEMAND_Tp12'... 'RG_TOTALDEMAND_Tp168',
        'RG_GEN_Solar_Tp6', 'RG_GEN_Solar_Tp12', ... 'RG_GEN_Solar_Tp168',
        'RG_GEN_Wind_Tp6', 'RG_GEN_Wind_Tp12', ... 'RG_GEN_Wind_Tp168',
    """
    output = {}
    ### First, we get predispatch data, which covers until the end of the next trading day (4am)
    filename = get_latest_file_from_folder('PredispatchIS_Reports')
    predis = pd.read_csv(get_subtable_from_file(filename, 'PREDISPATCH,REGION_SOLUTION'),
                        usecols=['REGIONID', 'DATETIME', 'AVAILABLEGENERATION', 'TOTALDEMAND', 
                                'NETINTERCHANGE', 'SS_SOLAR_UIGF', 'SS_WIND_UIGF'],
                        parse_dates=['DATETIME'])

    # IC_NET is defined to be NETINTERCHANGE * -1. IC_NET: import is +ve generation... though ignoring IC_net for now
    predis['NETINTERCHANGE'] = predis['NETINTERCHANGE'] * -1
    predis = predis.rename(columns={
        'NETINTERCHANGE': 'IC_NET',
        'SS_SOLAR_UIGF': 'GEN_Solar', 
        'SS_WIND_UIGF': 'GEN_Wind', 
    })

    # remove duplicates because one day there were multiples for some reason. It happened with PUBLIC_PREDISPATCHIS_202211151030_20221115100251.CSV. 
    predis = predis.drop_duplicates(['DATETIME', 'REGIONID'], keep='last')

    predis = predis.pivot(index='DATETIME', columns='REGIONID', values=['AVAILABLEGENERATION', 'TOTALDEMAND', 'GEN_Solar', 'GEN_Wind'])
    predis.columns = [f'{region}_{col}' for col, region in predis.columns]


    ### Second, we get ST PASA data, which covers the remainder of the 7 days
    filename = get_latest_file_from_folder('Short_Term_PASA_Reports')
    stpasa = pd.read_csv(get_subtable_from_file(filename, 'STPASA,REGIONSOLUTION'),
                        usecols=['REGIONID', 'INTERVAL_DATETIME', 'AGGREGATECAPACITYAVAILABLE', 'DEMAND50', 
                                'SS_SOLAR_UIGF', 'SS_WIND_UIGF'],
                        parse_dates=['INTERVAL_DATETIME'],
                        )
    stpasa = stpasa.rename(columns={
        'INTERVAL_DATETIME': 'DATETIME',
        'DEMAND50': 'TOTALDEMAND',
        'AGGREGATECAPACITYAVAILABLE': 'AVAILABLEGENERATION',
        'SS_SOLAR_UIGF': 'GEN_Solar', 
        'SS_WIND_UIGF': 'GEN_Wind', 
    })
    stpasa = stpasa.drop_duplicates(['REGIONID', 'DATETIME'], keep='first')  # there are multiple 'runs' in here, eg "LOR" and "OUTAGE_LRC". "LOR" first
    stpasa = stpasa.pivot(index='DATETIME', columns='REGIONID', values=['AVAILABLEGENERATION', 'TOTALDEMAND', 'GEN_Solar', 'GEN_Wind'])
    stpasa.columns = [f'{region}_{col}' for col, region in stpasa.columns]

    # predis + stpasa = (usually) 7 days of forecast
    gen = pd.concat([predis, stpasa])

    # like done in dataset generator, take the max over each N=6hr increment.
    gen = gen.resample('6H', label='right', closed='right').max()

    # we don't always have enough data to go right out to 7 days. Worst case is just before the new PASA/predis are released at 1pm NEM time
    # extend the data we do have by copying the final 24h two more times to extend by 48h, which is more than enough (36h would also be enough)
    last_24h = gen[-4:].copy()  # at 6H intervals, 4 entries == 24h
    gen = pd.concat([gen, last_24h, last_24h])

    for i, lag in enumerate(list(range(6, 168+6, 6))):
        for col in gen.columns:
            output[f'{col}_Tp{lag}'] = gen.iloc[i][col]
    return output

def get_availability_by_fuel(duids):
    """Get availability features by fuel type.
    
    The most up to date availabilty by feul source (up-to-minute data is secret, not available)
    Also uses the "daily report", the most recent one, which comes out at 4am each day
    Using instantaneous data from 24 hours ago as best estimate for availablity now. Alternative
    was to use the most recent, which is 4am today. Line ball call, went for this. 
    """
    filename = get_latest_file_from_folder('Daily_Reports') 
    df = pd.read_csv(get_subtable_from_file(filename, ',DUNIT,,3'),
                     usecols=['SETTLEMENTDATE', 'DUID', 'AVAILABILITY', ],
                     parse_dates=['SETTLEMENTDATE']
                    )
    # drop duplicates because there were multipe runs in other files
    df = df.drop_duplicates(['SETTLEMENTDATE', 'DUID'], keep='last')
    df = df.set_index('SETTLEMENTDATE')

    df = df.join(duids, on='DUID')

    # drop any loads, only want generators
    df = df[df.GENSETTYPE == 'GENERATOR']

    # rows that have same time of day as now (date ignored). 
    df = df[df.index.time == pd.Timestamp.now(tz='Australia/Brisbane').round('5min').time()]
    df = df.groupby(['REGIONID', 'CO2E_ENERGY_SOURCE']).agg({
        'AVAILABILITY': np.sum,
    }).reset_index()

    # feature names are eg VIC1_AVAILABILITY_Coal
    df['name'] = df['REGIONID'] + '_AVAILABILITY_' + df['CO2E_ENERGY_SOURCE']
    df = df.set_index('name')

    output = {}
    for region in REGIONIDS:
        for fuel in ['Battery Storage', 'Coal', 'Gas', 'Hydro', 'Solar', 'Wind']:
            col = f'{region}_AVAILABILITY_{fuel}'
            output[col] = df.at[col, 'AVAILABILITY'] if col in df.index else 0
    return output

def get_generation_by_fuel(duids):
    """Gets generaetion features broken down by fuel for the current time
    
    Uses table  `DISPATCHSCADA` to get generation for every plant (DUID).
    Note that availability isn't provided because it's secret till next day. 
    We don't have access to the table used in dataset generator, DISPATCHLOAD.
    """
    filename = get_latest_file_from_folder('Dispatch_SCADA')
    df = pd.read_csv(filename, 
                     skiprows=[0, -1], # skip first and last rows
                     usecols=['DUID', 'SCADAVALUE'])
    df = df.drop_duplicates(['DUID'], keep='last')
    df = df.set_index('DUID')

    df = df.join(duids)

    # drop any loads, only want generators
    df = df[df.GENSETTYPE == 'GENERATOR']

    df = df.groupby(['REGIONID', 'CO2E_ENERGY_SOURCE']).agg({
        'SCADAVALUE': np.sum,
        # 'AVAILABILITY': np.sum,
    }).reset_index()

    # no negatives 
    df['SCADAVALUE'] = df['SCADAVALUE'].clip(lower=0)

    # feature names are eg VIC1_GEN_Coal
    df['name'] = df['REGIONID'] + '_GEN_' + df['CO2E_ENERGY_SOURCE']
    df = df.set_index('name')

    output = {}
    for region in REGIONIDS:
        for fuel in ['Battery Storage', 'Coal', 'Gas', 'Hydro', 'Solar', 'Wind']:
            col = f'{region}_GEN_{fuel}'
            output[col] = df.at[col, 'SCADAVALUE'] if col in df.index else 0
    return output

def get_rooftop_pv():
    """Gets features for current and forecast rooftop pv output."""
    # Current values
    filename = get_latest_file_from_folder('ROOFTOP_PV/ACTUAL')
    df = pd.read_csv(filename, 
                     skiprows=1,
                     skipfooter=1,
                     engine='python',
                     usecols=['REGIONID', 'POWER'])
    df = df.drop_duplicates(['REGIONID'], keep='last')
    df = df.set_index('REGIONID')

    current_rooftop = {}
    for region in REGIONIDS:
        current_rooftop[f'{region}_GEN_Rooftop'] = df.at[region, 'POWER'].astype(float)

    # Rooftop PV forecasts
    filename = get_latest_file_from_folder('ROOFTOP_PV/FORECAST')
    df = pd.read_csv(filename, 
                     skiprows=1,
                     skipfooter=1,
                     engine='python',
                     usecols=['INTERVAL_DATETIME', 'REGIONID', 'POWERPOE50'])

    # remove duplicates because one day there were multiples in another file for some reason. It happened with PUBLIC_PREDISPATCHIS_202211151030_20221115100251.CSV. 
    df = df.drop_duplicates(['INTERVAL_DATETIME', 'REGIONID'], keep='last')

    df = df.pivot(index='INTERVAL_DATETIME', columns='REGIONID', values='POWERPOE50')

    # like done in dataset generator, take the max over each 6hr increment.
    df = df.rolling(2 * 6).max()
    df = df[11::12]  # take every 12th line, starting from the 11th, which is the end of the first block of 6hrs

    forecast_rooftop = {}
    for region in REGIONIDS:
        for i, lag in enumerate(range(6, 168+6, 6)):
            forecast_rooftop[f'{region}_GEN_Rooftop_Tp{lag}'] = df.iloc[i][region].astype(float)

    return current_rooftop | forecast_rooftop 

if __name__ == '__main__':
    # This code is not hit by lambda. Local dev only. 

    # optional: clear cache for testing
    if os.path.exists(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv'):
        print("Deleting 'greenness_rooftop_30d_cache.csv'")
        os.remove(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv')

    features, weather, recent_prices, greenness_month = get_features()

    # # test_get_recent_data.pkl
    # base_time = pd.Timestamp.now(tz='Australia/Brisbane').round('H').tz_localize(tz=None)
    # import pickle
    # with open('test_get_recent_data.pkl', 'wb') as f:
    #     pickle.dump((base_time, features, recent_prices, greenness_month), f, pickle.HIGHEST_PROTOCOL)

    print("\nComplete. Results:")
    print(features)