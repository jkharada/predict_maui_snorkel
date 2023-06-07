import configparser
import requests
import pandas as pd
import numpy as np
from datetime import date
from bs4 import BeautifulSoup
import re

parser = configparser.ConfigParser()
_ = parser.read('snorkel.cfg')
mesowest_auth_key = parser.get('MesoWest', 'token')

def get_features():
    df_buoy = get_buoy_features()
    df_weather = get_weather_features()
    
    df_buoy = df_buoy.merge(df_weather, left_index=True, right_index=True)
    df_buoy.index = df_buoy.index + pd.Timedelta(days=2)
    # features need to be in order to work correctly
    return df_buoy[['51000_WDIR', '51000_WSPD', '51000_GST', '51000_WVHT', '51000_DPD',
       '51000_APD', '51000_MWD', '51000_PRES', '51000_ATMP', '51000_WTMP',
       '51001_WDIR', '51001_WSPD', '51001_GST', '51001_WVHT', '51001_DPD',
       '51001_APD', '51001_MWD', '51001_PRES', '51001_ATMP', '51001_WTMP',
       '51001_DEWP', '51002_WDIR', '51002_WSPD', '51002_GST', '51002_WVHT',
       '51002_DPD', '51002_APD', '51002_MWD', '51002_PRES', '51002_ATMP',
       '51002_WTMP', '51003_WDIR', '51003_WSPD', '51003_GST', '51003_WVHT',
       '51003_DPD', '51003_APD', '51003_MWD', '51003_PRES', '51003_WTMP',
       '51004_WDIR', '51004_WSPD', '51004_GST', '51004_WVHT', '51004_DPD',
       '51004_APD', '51004_MWD', '51004_PRES', '51004_ATMP', '51004_WTMP',
       '51101_WDIR', '51101_WSPD', '51101_GST', '51101_WVHT', '51101_DPD',
       '51101_APD', '51101_MWD', '51101_PRES', '51101_ATMP', '51101_WTMP',
       'klih1_WDIR', 'klih1_WSPD', 'klih1_GST', 'klih1_PRES', 'klih1_ATMP',
       'klih1_WTMP', 'kwhh1_WDIR', 'kwhh1_WSPD', 'kwhh1_GST', 'kwhh1_PRES',
       'kwhh1_ATMP', 'kwhh1_WTMP', 'PHNY_peak_wind_direction', 'PHNY_wind_gust', 
        'PHNY_peak_wind_speed', 'A1741_wind_speed', 'A1741_wind_direction',
        'COOPKMLH1_precip_accum_24_hour', 'COOPKBSH1_air_temp_low_24_hour', 'COOPKBSH1_air_temp_high_24_hour',
        'COOPKBSH1_air_temp', 'PHOG_wind_gust', 'KHKH1_precip_accum_24_hour', 'PHMK_peak_wind_direction',
        'PHMK_peak_wind_speed', 'PHMK_wind_gust', 'PKKH1_precip_accum_24_hour', 'PHNY_air_temp_high_24_hour',
        'PHNY_air_temp_low_24_hour', 'PHMK_air_temp_low_24_hour', 'PHMK_air_temp_high_24_hour', 'HNAH1_pressure',
        'HNAH1_air_temp', 'HNAH1_wind_speed', 'HNAH1_wind_gust', 'HNAH1_wind_direction', 'HNAH1_precip_accum_24_hour',
        'AR427_wind_direction', 'PHOG_air_temp_low_24_hour', 'PHOG_air_temp_high_24_hour', 'AR427_wind_gust',
        'AR427_precip_accum_24_hour', 'AR427_pressure', 'AR427_air_temp', 'AR427_wind_speed', 'PUKH1_precip_accum_24_hour',
        'KACH1_precip_accum_24_hour', 'P36_wind_direction', 'P36_wind_speed', 'P36_precip_accum_24_hour', 'P36_pressure',
        'P36_air_temp', 'P36_wind_gust', 'WUKH1_precip_accum_24_hour', 'KLFH1_wind_speed', 'KLFH1_air_temp',
        'KLFH1_peak_wind_speed', 'KLFH1_wind_gust', 'KLFH1_wind_direction', 'KLFH1_peak_wind_direction',
        'KLFH1_precip_accum_24_hour', 'PHMK_dew_point_temperature', 'WWKH1_precip_accum_24_hour', 'AP834_wind_direction',
        'PHNY_dew_point_temperature', 'PHNY_visibility', 'PHNY_wind_direction', 'PHNY_wind_speed', 'PHNY_air_temp',
        'PHNY_pressure', 'MLKH1_air_temp_high_24_hour', 'MLKH1_air_temp_low_24_hour', 'COOPKACH1_precip_accum_24_hour',
        'COOPLAHH1_precip_accum_24_hour', 'COOPHNAH1_precip_accum_24_hour', 'COOPMABH1_precip_accum_24_hour',
        'COOPKBSH1_precip_accum_24_hour', 'COOPWUKH1_precip_accum_24_hour', 'KPDH1_dew_point_temperature',
        'KPDH1_wind_speed', 'KPDH1_wind_gust', 'KPDH1_precip_accum_24_hour', 'KPDH1_peak_wind_direction',
        'KPDH1_wind_direction', 'KPDH1_peak_wind_speed', 'KPDH1_air_temp', 'AP834_wind_speed', 'AP834_precip_accum_24_hour',
        'AP834_dew_point_temperature', 'WCCH1_precip_accum_24_hour', 'AP834_wind_gust', 'AP834_air_temp', 'AP834_pressure', 
        'MLKH1_precip_accum_24_hour', 'MLKH1_peak_wind_direction', 'MLKH1_wind_direction', 'MLKH1_peak_wind_speed',
        'MLKH1_air_temp', 'MLKH1_dew_point_temperature', 'MLKH1_wind_gust', 'MLKH1_wind_speed', 'PHMK_visibility',
        'PHMK_pressure', 'ULUH1_precip_accum_24_hour', 'PHMK_precip_accum_24_hour', 'KPNH1_precip_accum_24_hour',
        'PHMK_wind_speed', 'PHMK_wind_direction', 'PHMK_air_temp', 'PHOG_air_temp', 'PHOG_pressure', 'PHOG_wind_direction',
        'PHOG_wind_speed']].sort_index()

def get_buoy_features():
    # get buoy features for each station
    feature_dict = get_buoy_feature_dict()
    
    start_date = f"{date.today() - pd.Timedelta(days=8):%Y-%m-%d}"
    end_date = f"{date.today()-pd.Timedelta(days=1):%Y-%m-%d}"
    
    df_buoy_features = pd.DataFrame(index=pd.date_range(start_date, end_date))
    
    for station, meas_list in feature_dict.items():
        df_buoy_features = pd.merge(df_buoy_features, get_mean_station_data(station, meas_list), left_index=True, right_index=True)
    
    df_median = pd.read_csv('buoy_features_median.csv', index_col='Unnamed: 0') ## CREATE THIS TABLE
    medians = df_median['0'].to_dict()
    df_buoy_features.fillna(medians, inplace=True)
    
    return df_buoy_features[['51000_WDIR', '51000_WSPD', '51000_GST', '51000_WVHT', '51000_DPD',
       '51000_APD', '51000_MWD', '51000_PRES', '51000_ATMP', '51000_WTMP',
       '51001_WDIR', '51001_WSPD', '51001_GST', '51001_WVHT', '51001_DPD',
       '51001_APD', '51001_MWD', '51001_PRES', '51001_ATMP', '51001_WTMP',
       '51001_DEWP', '51002_WDIR', '51002_WSPD', '51002_GST', '51002_WVHT',
       '51002_DPD', '51002_APD', '51002_MWD', '51002_PRES', '51002_ATMP',
       '51002_WTMP', '51003_WDIR', '51003_WSPD', '51003_GST', '51003_WVHT',
       '51003_DPD', '51003_APD', '51003_MWD', '51003_PRES', '51003_WTMP',
       '51004_WDIR', '51004_WSPD', '51004_GST', '51004_WVHT', '51004_DPD',
       '51004_APD', '51004_MWD', '51004_PRES', '51004_ATMP', '51004_WTMP',
       '51101_WDIR', '51101_WSPD', '51101_GST', '51101_WVHT', '51101_DPD',
       '51101_APD', '51101_MWD', '51101_PRES', '51101_ATMP', '51101_WTMP',
       'klih1_WDIR', 'klih1_WSPD', 'klih1_GST', 'klih1_PRES', 'klih1_ATMP',
       'klih1_WTMP', 'kwhh1_WDIR', 'kwhh1_WSPD', 'kwhh1_GST', 'kwhh1_PRES',
       'kwhh1_ATMP', 'kwhh1_WTMP']]

def get_mean_station_data(station, meas_list):
    """
    Pulls buoy data from most recent 45 days and returns a data frame 
    with the mean value of each element in meas_list over one day.
    
    Input: station name as string, meas_list is a list of measurements 
    used as features for that data station
    
    Output: Dataframe with the mean daily values of the measurements for the last 45 days
    
    To do: fill in empty values with median or interpolated_values
    """
    response = requests.get('https://www.ndbc.noaa.gov/data/realtime2/' + str(station).upper() +'.txt')
    data = [ [ i for i in row.split()] for row in response.text.split('\n') ]
    df_buoy = pd.DataFrame(data, columns=data[0])
    
    df_buoy = df_buoy.drop(index=[0,1])[:-1] #drop last empty row and first two rows with header names
    df_buoy.replace(['MM'], None, inplace=True) #MM means no data
    
    df_buoy.rename(columns = {'#YY': 'year', 'MM': 'month', 'DD': 'day'}, inplace=True)
    df_buoy['date'] = pd.to_datetime(df_buoy[['year','month','day']])
    df_buoy.set_index('date', inplace=True)
    
    df_buoy = df_buoy[meas_list].apply(pd.to_numeric)
    df_buoy_agg = df_buoy.groupby('date').mean()
    df_buoy_agg.rename(lambda x: station + '_' + x, axis='columns', inplace=True)
    
    return df_buoy_agg

def get_buoy_feature_dict():
    """
    Returns dictionary with station as keys and a list of measurements as values
    """
    buoy_features = list(pd.read_csv('buoy_data_cleaned.csv').columns)
    buoy_features.remove('date')
    buoy_stations = [ feature.split('_')[0] for feature in buoy_features ]
    buoy_stations = list(set(buoy_stations))
    feature_dict = {station: [] for station in buoy_stations}

    for feature in buoy_features:
        station, val, mean = feature.split('_')
        if not val in feature_dict[station]:
            feature_dict[station].append(val)
        
    return feature_dict


def get_scores():
    """
    Gets the last 7 days of snorkel scores from the snorkel store website. 
    Returns a dataframe with the column names:
            ['dates', 'nw_scores', 's_scores', 'k_scores']
    
    """

    url = "https://thesnorkelstore.com/maui-snorkeling-conditions-reports/"
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
              'referer': 'https://thesnorkelstore.com/'
              }
    response = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(response.text, features="html.parser")
    parent = soup.find('div', attrs={'class': 'rte'})
    soup_scores = parent.find_all('strong')
    soup_dates = parent.find_all('h2')

    nw_scores = re.findall(r"Northwest: (\d.\d)", str(soup_scores))
    s_scores = re.findall(r"South Shore: (\d.\d)", str(soup_scores))
    k_scores = re.findall(r"Ka’anapali: (\d.\d)", str(soup_scores))
    dates = re.findall(r"[MTWFS]\w+day, [JFMAJSOND]\w+ \d\d? 20\d\d", str(soup_dates))

    #df_actual_scores = pd.DataFrame({'Dates': dates[:7], 'Actual': nw_scores[:7], 's_scores': s_scores[:7], 'k_scores': k_scores[:7]})
    df_actual_scores = pd.DataFrame({'Dates': pd.date_range(date.today() - pd.Timedelta(days=6),date.today()), 'Actual': nw_scores[:7][::-1], 's_scores': s_scores[:7][::-1], 'k_scores': k_scores[:7][::-1]})
    df_actual_scores['Dates'] = df_actual_scores['Dates'].apply(pd.to_datetime)
    return df_actual_scores

def get_weather_features():
    start_date = f"{date.today() - pd.Timedelta(days=8):%Y%m%d}" + '1000'
    end_date = f"{date.today() - pd.Timedelta(days=1):%Y%m%d}" + "1000"
    
    df_features = list(pd.read_csv('cleaned_features_mesowest.csv', index_col='Date_Time').columns)
    feature_dict = get_mesowest_feature_dict()
    
    df_new = pd.DataFrame()
    
    for station, meas in feature_dict.items():
        df_new = pd.concat([df_new, get_mean_mesowest_data(station, meas)], axis=1)
        
    # Fill in NaNs, if they exist
    df_new.interpolate('linear', limit=2, inplace=True)
    df_median = pd.read_csv('mesowest_features_medians.csv', index_col=0)
    medians = df_median['0'].to_dict()
    df_new.fillna(medians, inplace=True)
    
    return df_new[df_features]

from datetime import date
from time import sleep

def get_mean_mesowest_data(station, meas_list):
    start_date = f"{date.today() - pd.Timedelta(days=8):%Y%m%d}" + '1000'
    end_date = f"{date.today() - pd.Timedelta(days=1):%Y%m%d}" + "1000"
    
    params = {
    'token': mesowest_auth_key,
    'stid': station,
    'start': start_date, # 10:00 to offset UTC time 
    'end': end_date, # 10:00 to offset UTC time
    'obtimezone': 'local',
    'showemptystations': 0,
    'units': 'english',
    'vars': meas_list + ['precip_accum_one_hour', 'precip_accum']
    }
    response = requests.get('https://api.synopticdata.com/v2/stations/timeseries', params=params)
    
    while response.status_code != 200:
        time.sleep(5)
        params = {
        'token': mesowest_auth_key,
        'stid': station,
        'start': start_date, # 10:00 to offset UTC time 
        'end': end_date, # 10:00 to offset UTC time
        'obtimezone': 'local',
        'showemptystations': 0,
        'units': 'english',
        'vars': meas_list + ['precip_accum_one_hour', 'precip_accum']
        }
        response = requests.get('https://api.synopticdata.com/v2/stations/timeseries', params=params)
        
    weather_data = response.json()
    df = pd.DataFrame(weather_data['STATION'][0]['OBSERVATIONS'])
    df['date_time'] = pd.to_datetime(df['date_time']).dt.date
    df.drop(df[df['date_time'] == date.today()].index, inplace=True)
    df.rename(lambda x: x.replace('_set_1d','') if 'set_1d' in x else x, axis=1, inplace=True)
    df.rename(lambda x: x.replace('_set_1','') if 'set_1' in x else x, axis=1, inplace=True)
    
    df_new = df.groupby('date_time').mean()
    
    df_new = df_new.loc[:, ~df_new.columns.duplicated()]
    
    if 'peak_wind_speed' in meas_list and 'wind_gust' in df.columns:
        if 'peak_wind_speed' in df.columns:
            df_new['peak_wind_speed'][df_new['peak_wind_speed'].isnull()] = df.groupby('date_time').max(numeric_only=True)['wind_gust']
        else:
            df_new['peak_wind_speed'] = df.groupby('date_time').max(numeric_only=True)['wind_gust']

    if 'peak_wind_direction'in meas_list:
        if 'peak_wind_direction'in df.columns:
            df_new['peak_wind_direction'][df_new['peak_wind_direction'].isnull()] = df_new['peak_wind_direction'][df_new['peak_wind_direction'].isnull()].combine_first(df[['date_time','wind_direction']].loc[df.groupby('date_time')['wind_gust'].idxmax().dropna()].set_index('date_time').rename({'wind_direction':'peak_wind_direction'}, axis=1).squeeze())
        else:
            df_new['peak_wind_direction'] = df[['date_time','wind_direction']].loc[df.groupby('date_time')['wind_gust'].idxmax().dropna()].set_index('date_time').rename({'wind_direction':'peak_wind_direction'}, axis=1).squeeze()

    if 'precip_accum' in df.columns and 'precip_accum_24_hour' in meas_list:
        df_new['precip_accum_24_hour'] = df.groupby('date_time')['precip_accum'].agg(max) - df.groupby('date_time')['precip_accum'].first()
    elif 'precip_accum_24_hour'in meas_list and 'precip_accum_one_hour' in df.columns:
        if 'precip_accum_24_hour' in df_new.columns:
            df_new['precip_accum_24_hour'][df_new['precip_accum_24_hour'].isnull()] = df.groupby('date_time').sum(numeric_only=True)['precip_accum_one_hour']
        else:
            df_new['precip_accum_24_hour'] = df.groupby('date_time').sum(numeric_only=True)['precip_accum_one_hour']


    df_new.rename(lambda x: station + "_" + x if 'date' not in x else x, axis=1, inplace=True)
    
    return df_new


def get_mesowest_feature_dict():
    """
    Returns dictionary with station as keys and a list of measurements as values
    """
    weather_features = list(pd.read_csv('cleaned_features_mesowest.csv', index_col='Date_Time').columns)
    weather_stations = [ feature.split('_')[0] for feature in weather_features ]
    weather_stations = list(set(weather_stations))
    feature_dict = {station: [] for station in weather_stations}

    for feature in weather_features:
        split_feats = feature.split('_')
#        if not val in feature_dict[station]:
        feature_dict[split_feats[0]].append('_'.join(split_feats[1:]))
        
    return feature_dict

import pickle

def update_results():
    
    try:
        with open('week_results.p', 'rb') as f:
            df_results = pickle.load(f)
    except FileNotFoundError:
        return get_results()
    
    if pd.Timestamp(date.today()) == df_results.iloc[-1]['Dates']:
        try:
            with open('day_results.p', 'rb') as f:
                nw_prediction = pickle.load(f)
        except FileNotFoundError:
            return get_results()
            
        return nw_prediction, df_results
    
    else:
        return get_results()
        

def get_results():
    
    with open('1_day_mesowest_model.p', 'rb') as f:
        model = pickle.load(f)
        
    df_x = get_features()
    df_results = get_scores()
    
    y_predict = model.predict(df_x)
    
    df_results['Prediction'] = y_predict[:7]
    nw_prediction = y_predict[-1]
    
    with open('week_results.p', 'wb') as f:
        pickle.dump(df_results, f)
    
    with open('day_results.p', 'wb') as f:
        pickle.dump(nw_prediction, f)
    
    return nw_prediction, df_results

