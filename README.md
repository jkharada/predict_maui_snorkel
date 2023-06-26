# predict_snorkel_maui
## [https://maui-snorkel-prediction.onrender.com/](https://maui-snorkel-prediction.onrender.com/)

## Simple description:
Predicts snorkeling conditions in the Northwest region of Maui (Napili, Kapalua, and Honolua) based on the previous day's weather conditions. This model is intended to be updated at midnight of the current day, so that early risers in the Eastern part of Maui can know whether to head out to the Northwest region of Maui without having to wait for the official report from the Snorkel Store.

## About:
TDI Data Science Fellowship capstone project for Jaye Harada
 
## How does it work?
Snorkeling conditions are predicted using a supervised ensemble machine learning model that has been trained on historical weather data from the MesoWest's Mesonet API, historical buoy data from NOAA, and previous snorkel reports from the [Snorkel Store](https://thesnorkelstore.com/maui-snorkeling-conditions-reports/) to predict the snorkeling conditions for the current day. Predictions are made using the previous day's weather and buoy data. 
