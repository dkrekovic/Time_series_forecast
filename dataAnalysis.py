from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd


plt.rcParams.update({'figure.figsize': (20,10), 'figure.dpi': 120})
df = pd.read_csv("data/weather_daily_avg.csv", parse_dates=['datetime'])

def plot_df(weather_daily_avg, x, y, title="", xlabel='Date', ylabel='Temperature', dpi=100):
    plt.figure(figsize=(20, 10), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.datetime, y=(df.t2m - 273.15), title='Temperatura na 2m 2018-2020')

# Trend
df['year'] = [d.year for d in df.datetime]
df['month'] = [d.strftime('%b') for d in df.datetime]
years = df['year'].unique()

# BOXPLOT
fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=80)
sns.boxplot(x='year', y='t2m', data=df, ax=axes[0])
sns.boxplot(x='month', y='t2m', data=df.loc[~df.year.isin([2018, 2020]), :])
axes[0].set_title('Year-wise Box Plot\n(trend)', fontsize=18);
axes[1].set_title('Month-wise Box Plot\n(sezonalnost)', fontsize=18)
plt.show()

#DEKOMPOZICIJA

# Aditivna Dekompozicija
result_add = seasonal_decompose(df['t2m'], model='additive', extrapolate_trend='freq', period=365)
plt.rcParams.update({'figure.figsize': (20, 10)})
result_add.plot()
plt.show()
warnings.simplefilter(action='ignore', category=FutureWarning)

#STACIONARNOST

y=df['t2m']

# ADF Test
result = adfuller(df.t2m.values, autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    print(f'\nKPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(y, regression='ct')


def plot_variables(timeseries, title):
    rolmean = pd.Series(timeseries).rolling(window=30).mean()
    rolstd = pd.Series(timeseries).rolling(window=30).std()

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label=title)
    ax.plot(rolmean, label='varijanca');
    ax.plot(rolstd, label='standardna devijacija');
    ax.legend()
    plt.show()

pd.options.display.float_format = '{:.8f}'.format

plot_variables(y.dropna(inplace=False), 'serija' )

y_lag =  y - y.shift(40)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(y_lag.iloc[50:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(y_lag.iloc[50:], lags=40, ax=ax2)
pyplot.show()
