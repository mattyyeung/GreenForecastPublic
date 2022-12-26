""" get_greenness.py calculated the greenness features for the forecaster. 

greenness mean-of-last-30 days features (eg VIC1_Greenness_Tm30d_Mean) need most of the heavy lifting
Current greenness is also calculated.
Requires duids as input and will reuse a bunch of files downloaded already by the forecaster
Inputs gathered:
- Again, use the "daily report" (also used by 30d price lags): (all 30 days available from one folder)
  - gen by fuel: This time we want a different subtable `DUNIT`, which has all the data by DUID. 
  - net interconnector flow: `DREGION` subtable
- Rooftop PV Actual (not forecast) data - in its own separate dataset (only 14 (?) days available from current folder, go to archive for the remanider)
"""

import requests
import pandas as pd
import numpy as np
import io
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup
import zipfile
from time import sleep
import logging 
import math

from aemo import get_duids, yield_files_from_folder_by_index, get_all_urls_from_folder_index, download_and_unzip_file, get_subtable_from_file

REGIONIDS = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
DOWNLOAD_CACHE_FOLDER = Path('/tmp/forecaster_cache')

logging.basicConfig(level=logging.INFO)


def get_greenness(duids):
    """Top level function for this module. Calculates and returns current greenness and greenness lags (averages)
    Also returns greenness_month, the full data for greenness over the last 30 days (excluding current day). 
    """
    Path(DOWNLOAD_CACHE_FOLDER).mkdir(parents=True, exist_ok=True)  # make sure download cache folder exists;

    logging.info('Get Greenness for the last 30d:')
    inputs = greenness_inputs(duids)

    greenness_month, gen_by_fuel_month = calculate_real_greenness(inputs)

    # Current greenness - most recent in the list
    output = greenness_month.iloc[-1].to_dict()

    # Interconnector greenness features
    ic_columns = [f"{region}_IC_Green_In" for region in REGIONIDS] + [f"{region}_IC_Fossil_In" for region in REGIONIDS]
    output = output | gen_by_fuel_month[ic_columns].iloc[-1].to_dict()

    output = output | get_greenness_lags(greenness_month)

    # Now, calculate rolling averages for last 3/7/30 days
    for region in REGIONIDS:
        now_minus_3d = greenness_month.index.max() - pd.Timedelta(3, 'D')
        output[f'{region}_Greenness_Tm3d_Mean'] = greenness_month[now_minus_3d:].mean()[f'{region}_Greenness']
        now_minus_7d = greenness_month.index.max() - pd.Timedelta(7, 'D')
        output[f'{region}_Greenness_Tm7d_Mean'] = greenness_month[now_minus_7d:].mean()[f'{region}_Greenness']
        output[f'{region}_Greenness_Tm30d_Mean'] = greenness_month.mean()[f'{region}_Greenness']

    return output, gen_by_fuel_month, greenness_month

def greenness_inputs(duids):
    """ Gather the inputs for greenness calculation from the last 30 days
    
    Inputs needed are: PV, generation by fuel, interconnector flows
    Done in two parts: daily reports cover most of it, but the most recent few hours until the most recent
    daily report (released at 4am each day) need to be done differently. 
    """
    # First, get PV data: (not from daily reports; this returns all data including today)
    greenness_rooftop = get_last_30d_of_pv_data()

    logging.info('Get last 30 days of daily reports:')
    filenames = yield_files_from_folder_by_index('Daily_Reports', range(-1, -31, -1))

    daily_report_data = []
    failures = 0
    for filename in filenames:
        gen_by_duid = get_gen_by_duid_for_one_day(filename, duids)
        greenness_inputs = calculate_greenness_raw_for_one_day(gen_by_duid, greenness_rooftop)
        if greenness_inputs is False:
            # function failed to get data. Perhaps because of occasional known issue that pv data has a gap. 
            failures += 1
            assert failures < 10, "Failed to get 10 days of greenness data out of 30. Stopping."
        else: 
            # normal case, continue
            daily_report_data.append(greenness_inputs.join(get_interconnectors(filename)))

        # reduce max ephemeral space (in /tmp) used by the lambda function by deleting these big .csvs as we go. 
        # this works because yield_files_from_folder_by_index is a generator. 
        os.remove(filename)

    # do the same thing but for today's data which comes from different places:
    gen_by_duid = get_gen_by_duid_today(duids)
    # rooftop data is released with more lag. cut off gen_by_duid to match. In one test, rooftop was 1h (2 rows) behind.
    # TODO: nov28 check this line doesn't cause problems if there is a row or two missing eg if rooftop is a couple bhind...
    gen_by_duid = gen_by_duid[gen_by_duid.index <= greenness_rooftop.index.max()] 
    today_data = calculate_greenness_raw_for_one_day(gen_by_duid, greenness_rooftop)
    today_data = today_data.join(get_interconnectors())

    # put it all together
    df = pd.concat(daily_report_data)
    today_data = today_data[today_data.index > df.index.max()] # ignore data from today that overlaps daily reports
    df = pd.concat([today_data, df]).sort_index()

    return df

def get_last_30d_of_pv_data():
    """Get last 30d of rooftop pvdata from aemo datafiles.
    starts by either loading from disk if available or otherwise grabs week-large chunks from the nem archive
    then tops up either with most recent data
    """
    # cache is causing bugs when run twice... not used in production so just remove this; always run get_recent_weeks_of_pv_data() instead.
    # if os.path.exists(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv'):
    #     logging.info("Found cache 'greenness_rooftop_30d_cache.csv' ")
    #     df = pd.read_csv(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv',
    #                  parse_dates=['SETTLEMENTDATE'],
    #                  index_col='SETTLEMENTDATE')
    #     # delete the last day of data because it might be incomplete
    #     df = df[df.index < df.index[-1].normalize()]
    # else:
    #     logging.info("No Cache; running get_recent_weeks_of_pv_data()")
    #     df = get_recent_weeks_of_pv_data()
    df = get_recent_weeks_of_pv_data()

    today = pd.Timestamp.now(tz='Australia/Brisbane').tz_localize(tz=None).floor('D') # just a date, no time

    # if the data in the cache is old for some reason, just refresh entirely.
    if today.date() - df.index[-1].date() > timedelta(days=10):
        logging.info(f"Cache is old; refresh it with get_recent_weeks_of_pv_data(). today.date()={today.date()} df.index[-1].date()={df.index[-1].date()}")
        df = get_recent_weeks_of_pv_data()
    
    # delete anything from today's date 
    df = df[df.index < today]

    df = df.sort_index()

    # now get any remaining recent days, starting with the first missing date:
    recent_days = get_recent_days_of_pv_data(df.index[-1].date() + timedelta(days=1), today)

    greenness_rooftop = pd.concat([df, recent_days]).sort_index()
    
    #drop duplicates
    greenness_rooftop = greenness_rooftop[~greenness_rooftop.index.duplicated(keep='first')]

    greenness_rooftop.to_csv(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv')
    return greenness_rooftop

def get_recent_weeks_of_pv_data():
    """ download the 4 most recent weeks of data from the archive folder and extract into a dataframe"""
    rows = []
    PV_CACHE_FOLDER = DOWNLOAD_CACHE_FOLDER / 'RooftopPVActual'
    if not os.path.exists(PV_CACHE_FOLDER):
        os.makedirs(PV_CACHE_FOLDER)
    for nth_most_recent in [-4, -3, -2, -1]:

        for retries in range(5):
            response = requests.get('http://nemweb.com.au/Reports/Archive/ROOFTOP_PV/ACTUAL/')
            if (len(response.text) > 2000): # predipatch table is 3100 or so
                break
            sleep(0.2)
        soup = BeautifulSoup(response.text, features='lxml')
        latest = soup.find_all('a')[nth_most_recent]['href']
        zip_url = f'http://nemweb.com.au{latest}'
        filename = zip_url.split('/')[-1]
        filepath = PV_CACHE_FOLDER / filename
        print(f"Downloading {zip_url}")
        # extract the .zip contents (which is more .zip files, extract and parse them too)
        response = requests.get(zip_url)

        with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
            # outer_zip contains a bunch more zip files (only). Read these into memory too. 
            for inner_zip_name in outer_zip.namelist():
                with zipfile.ZipFile(outer_zip.open(inner_zip_name)) as inner_zip:
                    # inner zip files contain only one file - blah.csv. Read this into memory. 
                    with inner_zip.open(inner_zip.namelist()[0]) as csv_file:
                        rows.append(extract_row_from_pv_file(csv_file))

    df = pd.DataFrame(rows).set_index('SETTLEMENTDATE')
    return df

def extract_row_from_pv_file(csv_file):
    """Get last 30d of rooftop data"""
    df = pd.read_csv(csv_file,
                     skiprows=1,
                     skipfooter=1,
                     engine='python',
                     usecols=['INTERVAL_DATETIME', 'REGIONID', 'POWER'],
                     parse_dates=['INTERVAL_DATETIME'])
    df = df.drop_duplicates(['INTERVAL_DATETIME', 'REGIONID'], keep='last')
    df = df.set_index('REGIONID')

    row = {f'{region}_GEN_Rooftop': df.at[region, 'POWER'] for region in REGIONIDS}
    row['SETTLEMENTDATE'] = df.at['VIC1', 'INTERVAL_DATETIME']
    return row

def get_recent_days_of_pv_data(from_date, to_date):
    """ Get a few days worth of data from the most recent NEM folder, where everything is 
    stored as a separate file per half hour. 

    Gets data from pandas timstamps from_date to to_date inclusive.
    This data is midnight-to-midnight, unlike the data we're joinign to which is 4am-4am, but we can assume there's 0 PV between midnight and 4am!!!  :D
    downloads all the files (one per half-hour) for the given day 
    """  
    # create a list of substrings (dates) that we will use to choose which urls to download
    d = from_date
    match_strings = []
    while d <= to_date.date():
        # print("get_PV_actual_for_one_day(" + d.strftime('%Y%m%d') + ")")
        # days.append(get_PV_actual_for_one_day(d))

        # There are both "satellite" and "measurement" files. Two measurements of same thing. Assume we want satellite. 
        match_strings.append(f"SATELLITE_{d.strftime('%Y%m%d')}")  # eg 'SATELLITE_20220813'
        d = d + timedelta(days=1)

    # the full index has a URL for each half-hour timestep, we only want from a subset of days.
    all_urls = get_all_urls_from_folder_index('ROOFTOP_PV/ACTUAL')
    urls = []
    for match_str in match_strings:
        urls = urls + [s for s in all_urls if match_str in s]

    filenames = [download_and_unzip_file(url) for url in urls]

    # one row (file) per half-hour timestamp
    rows = []
    for filename in filenames:
        df = pd.read_csv(filename, skiprows=1, parse_dates=['INTERVAL_DATETIME'])
        df = df.drop_duplicates(['INTERVAL_DATETIME', 'REGIONID'], keep='last')
        df = df.set_index('REGIONID')
        row = {f"{region}_GEN_Rooftop": df.at[region,'POWER'] for region in REGIONIDS}
        row['SETTLEMENTDATE'] = df.at['VIC1', 'INTERVAL_DATETIME']
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index('SETTLEMENTDATE')
    return df

def get_gen_by_duid_today(duids):
    """Gets generation from all DUIDs for 'today' (at least since the most recent daily report)"""

    # only getting every 3rd one to save time. still 4 per hour to average.
    # we only need enough to get back to 4am (wehn the most recent daily report was released) but currently taking more than that. 
    filenames = yield_files_from_folder_by_index('Dispatch_SCADA', range(-1, -12*26, -3))

    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename, 
                         skiprows=[0, -1], # skip first and last rows
                         usecols=['DUID', 'SCADAVALUE', 'SETTLEMENTDATE'],
                         parse_dates=['SETTLEMENTDATE'])
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.drop_duplicates(['SETTLEMENTDATE', 'DUID'], keep='last')
    df = df.set_index('SETTLEMENTDATE')
    df = df.rename(columns={'SCADAVALUE': 'MW'})

    df = df.join(duids, on='DUID')

    return df

def get_gen_by_duid_for_one_day(filename, duids):
    df = pd.read_csv(get_subtable_from_file(filename, ',DUNIT,,3'),
                     usecols=['SETTLEMENTDATE', 'DUID', 'TOTALCLEARED'],
                     parse_dates=['SETTLEMENTDATE'],
                    )
    df = df.drop_duplicates(['SETTLEMENTDATE', 'DUID'], keep='last')
    df = df.set_index('SETTLEMENTDATE')
    df = df.rename(columns={'TOTALCLEARED': 'MW'})
    
    df = df.join(duids, on='DUID')
    
    return df

def calculate_greenness_raw_for_one_day(gen_by_duid, greenness_rooftop):
    """assumes gen_by_duid is a dataframe with columns ['SETTLEMENTDATE', 'DUID', 'MW', 'CO2E_ENERGY_SOURCE']"""
    df = gen_by_duid.copy()
    
    # drop any loads, only want generators
    df = df[df.GENSETTYPE == 'GENERATOR']

    # no negatives (saw one once somewhere else, some solar unit in qld)
    df['MW'] = df['MW'].clip(lower=0)

    # Group by fuel type
    df = df.groupby(['SETTLEMENTDATE', 'REGIONID', 'CO2E_ENERGY_SOURCE']).agg({
        'MW': np.sum,
    }).reset_index()

    df = df.set_index('SETTLEMENTDATE')
    # feature names are eg VIC1_GEN_Coal
    df['name'] = df['REGIONID'] + '_GEN_' + df['CO2E_ENERGY_SOURCE']

    # remove duplicates because one day there were multiples in another file for some reason. It happened with PUBLIC_PREDISPATCHIS_202211151030_20221115100251.CSV. 
    df = df.reset_index()
    df = df.drop_duplicates(['SETTLEMENTDATE', 'name'], keep='last')

    df = df.pivot(index='SETTLEMENTDATE', columns=['name'], values='MW')

    # change from 5min to 30min frequency to match PV_forecast. 
    # SETTLEMENTDATE is end-of-period so have origin of 5min offset from hour and name the bucket after the half hour, add 25mins
    df = df.resample('30min', origin='2010-01-01 00:05:00').mean()
    df.index = df.index + pd.tseries.frequencies.to_offset('25min')  # because SETTLEMENTDATE is end-of-period

    # add in rooftop PV data
    df = df.join(greenness_rooftop)

    # we can assume 0 PV between midnight and 4am!!!  :D
    if df.isna().sum().sum() >= 70:
        # import IPython; IPython.embed() # drop into interpreter for debug 
        logging.info(df)
        logging.warning(f"Problem getting last 30d of greenness.")
        logging.warning(f"In calculate_greenness_raw_for_one_day({df.isna().index[0]})")
        logging.warning(f"Some NAs in PV data is ok because they're at night, but {df.isna().sum().sum()} is too many")
        logging.warning(f"Could be missing data in data source again. Check http://nemweb.com.au/Reports/CURRENT/ROOFTOP_PV/ACTUAL/ and http://nemweb.com.au/Reports/Archive/ROOFTOP_PV/ACTUAL/")
        logging.warning(f"Attempting to continue by skipping this day...")
        return False

    df = df.fillna(0)

    for region in REGIONIDS:
        all_gen = [f'{region}_GEN_{fuel}' for fuel in ['Coal', 'Gas', 'Hydro', 'Solar', 'Wind', 'Rooftop']]
        renewable = [f'{region}_GEN_{fuel}' for fuel in ['Hydro', 'Solar', 'Wind', 'Rooftop']]

        # add in any missing data, eg SA coal, set to zeros. 
        for feature in all_gen:
            if feature not in df.columns:
                df[feature] = 0

        df[f'{region}_Greenness_raw'] = df[renewable].sum(axis=1) / df[all_gen].sum(axis=1)            
    
    return df
    
# def get_ics_today():
#     # Get interconnector net for ~='today', ie since the last daily report was released (4amish)
#     # only getting every 3rd one to save time. still 4 per hour to average.
#     # we only need enough to get back to 4am (wehn the most recent daily report was released) but currently taking more than that. 
#     filenames = yield_files_from_folder_by_index('DispatchIS_Reports', range(-1, -12*26, -3))
#     dfs = []
#     for filename in filenames:
#         df = pd.read_csv(get_subtable_from_file(filename, 'DISPATCH,REGIONSUM,'),
#                          usecols=['REGIONID', 'SETTLEMENTDATE', 'NETINTERCHANGE'])
#         dfs.append(df)

#     df = pd.concat(dfs)

#     # drop duplicates because there can be multiple different runs
#     df = df.drop_duplicates(['SETTLEMENTDATE', 'REGIONID'], keep='last')

#     # IC_NET is defined to be NETINTERCHANGE * -1. IC_NET: import is +ve generation.
#     df['NETINTERCHANGE'] = df['NETINTERCHANGE'] * -1

#     df = df.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='NETINTERCHANGE')

#     df.columns = [f'{region}_IC_NET' for region in df.columns]

#     # change from 5min to 30min frequency to match PV_forecast
#     df = df.resample('30min', origin='start').mean()
#     df.index = df.index + pd.tseries.frequencies.to_offset('25min')  # because SETTLEMENTDATE is end-of-period

def get_interconnectors(daily_report_filename=None):
    """ Get interconnector net for one day

    If daily_report_filename is specified, grab from that.
    Otherwise, get for ~='today', ie since the last daily report was released (4amish). This is from different files
    """
    if daily_report_filename:
        # get from a daily report
        df = pd.read_csv(get_subtable_from_file(daily_report_filename, ',DREGION,,3'),
                         usecols=['REGIONID', 'NETINTERCHANGE', 'SETTLEMENTDATE', ],
                         parse_dates=['SETTLEMENTDATE']
                        )
    else:
        # only getting every 3rd one to save time. still 4 per hour to average.
        # we only need enough to get back to 4am (wehn the most recent daily report was released) but currently taking more than that. 
        filenames = yield_files_from_folder_by_index('DispatchIS_Reports', range(-1, -12*26, -3))
        dfs = []
        for filename in filenames:
            df = pd.read_csv(get_subtable_from_file(filename, 'DISPATCH,REGIONSUM,'),
                             usecols=['REGIONID', 'SETTLEMENTDATE', 'NETINTERCHANGE'],
                             parse_dates=['SETTLEMENTDATE'])
            dfs.append(df)

        df = pd.concat(dfs)

    # remove duplicates because one day there were multiples in another file for some reason. It happened with PUBLIC_PREDISPATCHIS_202211151030_20221115100251.CSV. 
    df = df.drop_duplicates(['SETTLEMENTDATE', 'REGIONID'], keep='last')

    # IC_NET is defined to be NETINTERCHANGE * -1. IC_NET: import is +ve generation.
    df['NETINTERCHANGE'] = df['NETINTERCHANGE'] * -1

    df = df.pivot(index='SETTLEMENTDATE', columns='REGIONID', values='NETINTERCHANGE')

    df.columns = [f'{region}_IC_NET' for region in df.columns]

    # change from 5min to 30min frequency to match PV_forecast. 
    # SETTLEMENTDATE is end-of-period so have origin of 5min offset from hour and name the bucket after the half hour, add 25mins
    df = df.resample('30min', origin='2010-01-01 00:05:00').mean()
    df.index = df.index + pd.tseries.frequencies.to_offset('25min')

    return df

def get_greenness_lags(greenness_month):
    lags = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 168]

    now = (pd.Timestamp.utcnow()
           .tz_convert('Australia/Brisbane')  # Use Brisbane tz because it's always AEST == NEM time.
           .floor('H')
           .tz_localize(tz=None))  # remove timezone

    output = {}
    for region in REGIONIDS:
        output = output | {f'{region}_Greenness_Tm{lag}': greenness_month[f'{region}_Greenness'][now - pd.Timedelta(lag, 'H')] for lag in lags}
    return output

def calculate_real_greenness(greenness_inputs):
    """ calculates final greenness values, also returns IC_Green and IC_Fossil which are generated on the way
    """
    greenness = pd.DataFrame(index=greenness_inputs.index)

    ############
    ### Copied from dataset generator. Consider refactoring if making changes. 

    # Now, un-raw it by adding in interconnector effects
    # calcaulte "IC_Import" which is all interconnector imports, ignoring any exports
    # This is because any exports won't change the greenness for this state
    # (this is a bit of an approximation? but close enough since IC flows don't dominate vic/nsw)
    # Then, calculate the amount of green energy flowing in, which is IC_import * greenness_raw from the incoming state.
    # this is complex for VIC & NSW, easier for other 3 states with only one feeder state. 
    # with IC_Import and IC_Green_In, we can then calculate the greenness.
    gi = greenness_inputs

    # QLD/SA/TAS are simpler
    gi['QLD1_IC_Import'] = gi['QLD1_IC_NET'].clip(lower=0)
    gi['SA1_IC_Import'] = gi['SA1_IC_NET'].clip(lower=0)
    gi['TAS1_IC_Import'] = gi['TAS1_IC_NET'].clip(lower=0)

    gi['QLD1_IC_Green_In'] = gi['QLD1_IC_Import'] * gi['NSW1_Greenness_raw']
    gi['SA1_IC_Green_In'] = gi['SA1_IC_Import'] * gi['VIC1_Greenness_raw']
    gi['TAS1_IC_Green_In'] = gi['TAS1_IC_Import'] * gi['VIC1_Greenness_raw']

    # NSW1: 
    gi['NSW1_Import_From_Qld'] = gi['QLD1_IC_NET'].clip(upper=0) * -1
    gi['NSW1_IC_Net_From_Vic'] = gi['NSW1_IC_NET'] - (-1) * gi['QLD1_IC_NET']
    gi['NSW1_Import_From_Vic'] = gi['NSW1_IC_Net_From_Vic'].clip(lower=0)
    gi['NSW1_IC_Import'] = gi['NSW1_Import_From_Qld'] + gi['NSW1_Import_From_Vic']
    gi['NSW1_IC_Green_In'] = (gi['NSW1_Import_From_Qld'] * gi['QLD1_Greenness_raw']+ 
                              gi['NSW1_Import_From_Vic'] * gi['VIC1_Greenness_raw'])

    # VIC1:
    gi['VIC1_Import_From_Tas'] = gi['TAS1_IC_NET'].clip(upper=0) * -1
    gi['VIC1_Import_From_SA']  = gi['SA1_IC_NET'].clip(upper=0) * -1
    gi['VIC1_Import_From_NSW'] = gi['NSW1_IC_Net_From_Vic'].clip(upper=0) * -1
    gi['VIC1_IC_Import'] = gi['VIC1_Import_From_Tas'] + gi['VIC1_Import_From_SA'] + gi['VIC1_Import_From_NSW']
    gi['VIC1_IC_Green_In'] = (gi['VIC1_Import_From_Tas'] * gi['TAS1_Greenness_raw'] +
                              gi['VIC1_Import_From_SA']  * gi['SA1_Greenness_raw'] +
                              gi['VIC1_Import_From_NSW'] * gi['NSW1_Greenness_raw'])

    # create _IC_Fossil_In columns
    for region in REGIONIDS:
        gi[f'{region}_IC_Fossil_In'] = (gi[f'{region}_IC_Import'] - gi[f'{region}_IC_Green_In'])

    # calculate greenness for real now we have IC_Import and IC_Green_In
    for region in REGIONIDS:
        # greenness_inputs[f'{region}_IC_Import'].plot(figsize=(16,5))

        renewable = [f'{region}_GEN_{fuel}' for fuel in ['Hydro', 'Solar', 'Wind', 'Rooftop']] + [f'{region}_IC_Green_In']
        all_gen = [f'{region}_GEN_{fuel}' for fuel in ['Coal', 'Gas', 'Hydro', 'Solar', 'Wind', 'Rooftop']] + [f'{region}_IC_Green_In', f'{region}_IC_Fossil_In']

        # convert to nparray with dtype=object to avoid a warning being thrown
        all_gen = np.array(all_gen, dtype=object)
        renewable = np.array(renewable, dtype=object)

        # final greenness calc - express as a percentage not [0,1]
        greenness[f'{region}_Greenness'] = gi[renewable].sum(axis=1) / gi[all_gen].sum(axis=1) * 100

    
    # all_gen features for all regions:
    gen_by_fuel = gi[[col for col in gi.columns if '_GEN_' in col or '_IC_Green_In' in col or '_IC_Fossil_In' in col]]
    
    greenness.index.name = "SETTLEMENTDATE"

    greenness = greenness.round(3)
    gen_by_fuel = gen_by_fuel.round(3)

    return greenness, gen_by_fuel


if __name__ == '__main__':
    # This code is not hit by lambda. Local dev only. 

    # optional: clear cache for testing
    if os.path.exists(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv'):
        print("Deleting 'greenness_rooftop_30d_cache.csv'")
        os.remove(DOWNLOAD_CACHE_FOLDER / 'greenness_rooftop_30d_cache.csv')


    get_greenness(get_duids())