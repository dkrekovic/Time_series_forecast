import numpy as np
import statsmodels.api as sm
import itertools
import pandas as pd
import matplotlib.pyplot as plt

weather = pd.read_csv('data/allt2m.csv', index_col='datetime', parse_dates=True)

weather_daily_avg = (weather.resample('24H').mean() - 273.15)
monthly_avg=weather_daily_avg.resample('M').mean()

weather_daily_avg.head()

y = weather_daily_avg['t2m']
y_to_train = y[:'2019-12-31'] # skup za treniranje
y_to_val = y['2020-01-01':] # skup za testtiranje
predict_date = len(y) - len(y[:'2020-01-01'])

def sarima_grid_search(y, seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]
    mini = float('+inf')
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()

                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini,
                                                                                       mini))
sarima_grid_search(y, 30)

def sarima_eva(y, order, seasonal_order, seasonal_period, pred_date, y_to_test):
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,
                                                                                                      round(
                                                                                                          np.sqrt(mse),
                                                                                                          2)))

    ax = y.plot(label='Real value')
    y_forecasted.plot(ax=ax, label='Forecast one step ahead', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)


    plt.legend()
    plt.show()

    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,
                                                                                                     round(np.sqrt(
                                                                                                         mse_dynamic),
                                                                                                           2)))

    ax = y.plot(label='Real value')
    y_forecasted_dynamic.plot(label='Forecast', ax=ax, figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)


    ax.set_ylabel('Temperature')

    plt.legend()
    plt.show()

    return (results)
model = sarima_eva(y,(1, 0, 1),(0, 1, 1, 30),30,'2020-01-01',y_to_val)


def forecast(model, predict_steps, y):
    pred_uc = model.get_forecast(steps=predict_steps)

    pred_ci = pred_uc.conf_int()

    ax = y.plot(label='Real value', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_ylabel(y.name)

    plt.legend()
    plt.show()

    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date', 'Forecast']
    pci = pred_ci.reset_index()
    pci.columns = ['Date', 'Lower Bound', 'Upper Bound']
    final_table = pm.join(pci.set_index('Date'), on='Date')

    return (final_table)

final_table = forecast(model,30,y)
final_table.head()

