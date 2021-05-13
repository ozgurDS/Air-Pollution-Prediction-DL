# Air-Pollution-Prediction-with-Deep-Learning

This repository inclludes a deep learning model for predicting the air pollution value for the next hour based on the given input. The dataset contains reports on the weather and the level of pollution each hour for five years at the US embassy in Beijing, China.

The data includes the date-time, the pollution called PM2.5 concentration, and the weather information including dew point, temperature, pressure, wind direction, wind speed and the cumulative number of hours of snow and rain. The complete feature list in the raw data is as follows:

No: row number
year: year of data in this row
month: month of data in this row
day: day of data in this row
hour: hour of data in this row
pm2.5: PM2.5 concentration
DEWP: Dew Point
TEMP: Temperature
PRES: Pressure
cbwd: Combined wind direction
Iws: Cumulated wind speed
Is: Cumulated hours of snow
Ir: Cumulated hours of rain

We used this data and frame a forecasting problem where, given the weather conditions and pollution for prior hours, we forecast the pollution at the next hour.

The RMSE (Root Mean Square Error) value of the model calculated as 24 while the mean value of the pollution was 98. RMSE punishes large errors more, so while the model performed pretty good with the general of the data, it naturally struggled more with the outline data and this is the main factor that increases the RMSE. In conclution, we can safely say that the model performs pretty good in general.
