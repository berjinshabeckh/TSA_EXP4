# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 
# Name: H.Berjin Shabeck
# Reg no: 212222240018



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd

# Dataset
df = pd.read_csv('score.csv')

# Step 1: Import necessary libraries
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set up the plot size
plt.rcParams['figure.figsize'] = [10, 6]

# Step 2: Define ARMA(1,1) process and generate data
ar1 = np.array([1, -0.5])  # AR coefficient
ma1 = np.array([1, 0.5])   # MA coefficient
arma11_process = ArmaProcess(ar1, ma1)
simulated_data_arma11 = arma11_process.generate_sample(nsample=1000)

# Step 3: Plot ARMA(1,1) time series
plt.figure()
plt.plot(simulated_data_arma11)
plt.title('Simulated ARMA(1,1) Time Series')
plt.xlim([0, 200])  # Limiting x-axis for better visualization
plt.show()

# Step 4: Display ACF and PACF for ARMA(1,1)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(simulated_data_arma11, lags=40, ax=axes[0])
plot_pacf(simulated_data_arma11, lags=40, ax=axes[1])
plt.show()

# Step 5: Define ARMA(2,2) process and generate data
ar2 = np.array([1, -0.5, 0.25])  # AR coefficients
ma2 = np.array([1, 0.5, 0.25])   # MA coefficients
arma22_process = ArmaProcess(ar2, ma2)
simulated_data_arma22 = arma22_process.generate_sample(nsample=10000)

# Step 6: Plot ARMA(2,2) time series
plt.figure()
plt.plot(simulated_data_arma22)
plt.title('Simulated ARMA(2,2) Time Series')
plt.xlim([0, 200])  # Limiting x-axis for better visualization
plt.show()

# Step 7: Display ACF and PACF for ARMA(2,2)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(simulated_data_arma22, lags=40, ax=axes[0])
plot_pacf(simulated_data_arma22, lags=40, ax=axes[1])
plt.show()

# Step 8: Fit ARMA model to the Scores dataset

# Check if the data is stationary (required for ARMA)
adf_test = sm.tsa.adfuller(df['Scores'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')

# Fit ARMA(1,1) to the Scores data
# Use statsmodels.tsa.arima.model.ARIMA with order (p,d,q)
# For ARMA(1,1) use order (1,0,1) 
arma_model_11 = sm.tsa.arima.model.ARIMA(df['Scores'], order=(1,0,1)).fit()
print(arma_model_11.summary())

# Fit ARMA(2,2) to the Scores data
# Use statsmodels.tsa.arima.model.ARIMA with order (p,d,q)
# For ARMA(2,2) use order (2,0,2) 
arma_model_22 = sm.tsa.arima.model.ARIMA
```

### OUTPUT:
![download](https://github.com/user-attachments/assets/1b24a9b0-59b3-4dae-87eb-c0bc3544004b)
![download](https://github.com/user-attachments/assets/ff93cfe0-827d-49d6-b04c-dd8b2f0b8b2c)
![download](https://github.com/user-attachments/assets/08fa20d9-6bf8-4357-969f-70532c7cf7b4)
![download](https://github.com/user-attachments/assets/b4bf57b0-f74e-4c08-85c1-7a27f7591681)
ADF Statistic: -1.1347911485458624
p-value: 0.7010001236817096

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
