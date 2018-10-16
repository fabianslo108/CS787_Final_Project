# Necessary libraries are loaded
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import Series
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

data_file = Path(r'C:\Users\Fabian\Dropbox\CS 767\Project\labor-conflict-data.csv')
data = pd.read_csv(filepath_or_buffer=data_file)
data = data.pivot(index='MonthYear', columns='QuadClass', values='f0_')
rng = pd.date_range(start='1/1/1999', periods=data.shape[0], freq='M')     # date index
data.index = rng
data.columns = ['Verbal_Coop', 'Material_Coop', 'Verbal_Conflict', 'Material_Conflict']
print(data.head())

data.plot()
plt.show()

output = Series(data['Material_Conflict'])[12:]
print(output)

input = pd.concat([data.shift(1), data.shift(2), data.shift(3), data.shift(4), data.shift(5), data.shift(6),
                  data.shift(7), data.shift(8), data.shift(9), data.shift(10), data.shift(11), data.shift(12)], axis=1).dropna()
print(input.head())


plot_acf(Series(data['Material_Conflict']), lags=50)
plt.show()

plot_pacf(Series(data['Material_Conflict']), lags=50)
plt.show()

# fit model
n_fit = data['Material_Conflict'].shape[0]//3*2
model = ARIMA(Series(data['Material_Conflict'])[:n_fit], order=(2,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
#plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


X = Series(data['Material_Conflict']).values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = np.square(np.subtract(test, predictions)).mean()
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
