#Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

#Load data 
data_full = pd.read_csv('/Users/Yaroslav/Downloads/Serbia_materials.csv')

#Create a dataset to append your forecasts to
forecast_2021_2030 = pd.DataFrame(columns=['CMG','Forecast_2030'])

#Loop through your CMG data
for CMG in data_full['CMG'].unique():
    #Filter and prepare the training data
    train = data_full[data_full['CMG']==CMG]
    russia_quarter = train['Volume'].tolist()
    russia_index_quarter = pd.date_range(start="2018-Q1", end="2021-Q3", freq="QS-OCT")
    russia_quarter_dataset = pd.Series(russia_quarter, russia_index_quarter)    
    #Fit the model
    russia_quarter_expo = ExponentialSmoothing(
        russia_quarter_dataset,
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        use_boxcox=False,
        damped_trend=True,
        initialization_method="estimated",
    ).fit(remove_bias=True)
    #Make your prediction
    russia_quarter_expo_forecast = russia_quarter_expo.forecast(37)
    russia_quarter_expo_forecast[np.isnan(russia_quarter_expo_forecast)] = 1
    print(russia_quarter_expo_forecast[33]+russia_quarter_expo_forecast[34]+russia_quarter_expo_forecast[35]+russia_quarter_expo_forecast[36])
    #Calculate value for 2030 and append it into your dataset
    forecast_2030_new = round(russia_quarter_expo_forecast[33]+russia_quarter_expo_forecast[34]+russia_quarter_expo_forecast[35]+russia_quarter_expo_forecast[36])
    forecast_2021_2030_new = pd.DataFrame([[CMG,forecast_2030_new]], columns=['CMG','Forecast_2030'])
    forecast_2021_2030 = forecast_2021_2030.append(forecast_2021_2030_new, ignore_index = True)
#Save the forecast into csv
forecast_2021_2030.to_csv('Serbia_materials_forecast_ets_2030_extraction.csv')