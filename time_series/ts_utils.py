
## for data
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, date
from dotenv import load_dotenv

## for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches

## for stationarity test
import statsmodels.api as sm

## for outliers detection, models tuning, clustering
from sklearn import preprocessing, svm, model_selection, metrics, cluster

## for autoregressive models
import pmdarima
import statsmodels.tsa.api as smt
import arch

import tensorflow as tf
## for deep learning
from tensorflow.python.keras import models, layers, preprocessing as kprocessing

## for prophet
from fbprophet import Prophet
pd.plotting.register_matplotlib_converters()

## for parametric fit and resistence/support
from scipy import optimize, stats, signal, cluster as sci_cluster

## for clustering
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################

def get_data_api_toTs(ini,coin):
    coin_url = os.getenv(coin.upper()+"_HISTOHOUR")
    if ini == 0 :
         request = requests.get(coin_url)
    else:
        request = requests.get(coin_url+f"&toTs={ini}")
    todo = json.loads(request.content)
    return todo['Data']['Data']

def convertToDF(dfJSON):
    return(pd.json_normalize(dfJSON))

'''
get cryptocurrency dataSet
:parameter
    :param coin: coin name (BTC,ETH or XRP)
    :param researches: number of observations * 2001
'''

def get_data_df(coin,researches):
    load_dotenv()
    data = get_data_api_toTs(0,coin)
    df_aux = convertToDF(data)
    for x in range(researches-1):
        ini = df_aux['time'][0]
        print("Buscando dados de : ",datetime.fromtimestamp(ini))
        data1=get_data_api_toTs(ini,coin)
        df_aux1 = convertToDF(data1)
        df_aux = df_aux1.append(df_aux,ignore_index=True)
    return df_aux
    
'''
get cryptocurrency dataSet
:parameter
    :param coin: coin name (BTC,ETH or XRP)
    :param sample_data: get sample data from api? (true or false)
'''
def get_data(coin, sample_data=True):

    if coin.upper() not in ('BTC', 'ETH', 'XRP'):
        err_msg = coin + ' is a invalid coin!'
        raise ValueError(err_msg)

    name_coin = "_SAMPLE_DATA" if sample_data else "_ALL_DATA"
    name_coin = coin.upper() + name_coin

    print("\nBuscando ", "amostra" if sample_data else "todas",
        " observações da moeda", coin.upper())

    load_dotenv()

    coin_url = os.getenv(name_coin)

    request = requests.get(coin_url)
    data = json.loads(request.content)
    content = data.get("Data")
    content = content.get("Data")

    print("Dataset foi carregado! Formatando Dataset ...")

    df = pd.json_normalize(content[0])

    for i in range(1, len(content)):
        observation = content[i]  

        df_temp = pd.json_normalize(observation)

        df = pd.DataFrame.append(df, df_temp)

    return df

'''
Plot ts with rolling mean and 95% confidence interval with rolling std.
:parameter
    :param ts: pandas Series
    :param window: num for rolling stats
    :param plot_intervals: bool - if True plots the conf interval
    :param plot_ma: bool - if True plots the moving avg
'''


def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(15,5)):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(ts.name)
    plt.plot(ts[window:], label='ts', color="black")
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window), color="red")
    if plot_intervals:
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound, color='lightskyblue', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



'''
Fit a parametric trend line.
:parameter
    :param ts: pandas Series
    :param degree: polynomial order, ex. if 1 --> trend line = constant + slope*x, if 2 --> trend line = constant + a*x + b*x^2
'''
def fit_trend(ts, degree=1, plot=True, figsize=(15,5)):
    ## fit trend
    dtf = ts.to_frame(name="ts")
    params = np.polyfit(ts.reset_index().index, ts.values, deg=degree)
    costant = params[-1]    
    dtf["trend"] = costant
    X = np.array(range(1,len(ts)+1))
    for i in range(1,degree+1):
        dtf["trend"] = dtf["trend"] + params[i-1]*(X**i)
        
    ## plot
    if plot is True:
        ax = dtf.plot(grid=True, title="Fitting Trend", figsize=figsize, color=["black","red"])
        ax.set(xlabel=None)
        plt.show()
    return dtf, params
        

'''
Defferenciate ts.
:parameter
    :param ts: pandas Series
    :param lag: num - diff[t] = y[t] - y[t-lag]
    :param order: num - how many times it has to differenciate: diff[t]^order = diff[t] - diff[t-lag] 
    :param drop_na: logic - if True Na are dropped, else are filled with last observation
'''
def diff_ts(ts, lag=1, order=1, drop_na=True):
    for i in range(order):
        ts = ts - ts.shift(lag)
    ts = ts[(pd.notnull(ts))] if drop_na is True else ts.fillna(method="bfill")
    return ts



'''
Find outliers using sklearn unsupervised support vetcor machine.
:parameter
    :param ts: pandas Series
    :param perc: float - percentage of outliers to look for
:return
    dtf with raw ts, outlier 1/0 (yes/no), numeric index
'''
def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda x: 1 if x == -1 else 0)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found "+str(sum(dtf_outliers["outlier"] == 1)))
    ax.plot(dtf_outliers.index, dtf_outliers["ts"], color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1].index, y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'], color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers



'''
Interpolate outliers in a ts.
'''
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5, title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black", label="interpolated", legend=True)
    ax.set(xlabel=None)
    plt.show()
    return ts_clean



'''
Finds Maxs, Mins, Resistence and Support levels.
:parameter
    :param ts: pandas Series
    :param window: int - rolling window
    :param trend: bool - False if ts is flat
:return
    dtf with raw ts, max, min, resistence, support
'''
def resistence_support(ts, window=30, trend=False, plot=True, figsize=(15,5)):
    dtf = ts.to_frame(name="ts")
    dtf["max"], dtf["min"] = [np.nan, np.nan]
    rolling = dtf['ts'].rolling(window=window).mean().dropna()  

    ## maxs
    local_max = signal.argrelextrema(rolling.values, np.greater)[0]
    local_max_idx = [dtf.iloc[i-window:i+window]['ts'].idxmax() for i in local_max if (i > window) and (i < len(dtf)-window)]
    dtf["max"].loc[local_max_idx] = dtf["ts"].loc[local_max_idx]

    ## mins
    local_min = signal.argrelextrema(rolling.values, np.less)[0]
    local_min_idx = [dtf.iloc[i-window:i+window]['ts'].idxmin() for i in local_min if (i > window) and (i < len(dtf)-window)]
    dtf["min"].loc[local_min_idx] = dtf["ts"].loc[local_min_idx]

    ## resistence/support
    dtf["resistence"] = dtf["max"].interpolate(method="linear") if trend is True else dtf["max"].fillna(method="ffill")
    dtf["support"] = dtf["min"].interpolate(method="linear") if trend is True else dtf["min"].fillna(method="ffill")
    
    ## plot
    if plot is True:
        ax = dtf["ts"].plot(color="black", figsize=figsize, grid=True, title="Resistence and Support")
        dtf["resistence"].plot(ax=ax, color="darkviolet", label="resistence", grid=True, linestyle="--")
        dtf["support"].plot(ax=ax, color="green", label="support", grid=True, linestyle="--")
        ax.scatter(x=dtf["max"].index, y=dtf["max"].values, color="darkviolet", label="max")
        ax.scatter(x=dtf["min"].index, y=dtf["min"].values, color="green", label="min")
        ax.set(xlabel=None)
        ax.legend()
        plt.show()
    return dtf



###############################################################################
#                 MODEL DESIGN & TESTING - FORECASTING                        #
###############################################################################
'''
Split train/test from any given data point.
:parameter
    :param ts: pandas Series
    :param exog: array len(ts) x n regressors
    :param test: num or str - test size (ex. 0.20) or index position (ex. "yyyy-mm-dd", 1000)
:return
    ts_train, ts_test, exog_train, exog_test
'''
def split_train_test(ts, exog=None, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|", ts.index[split], "| test size:", perc, " ---")
    
    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=figsize)
        ts_train.plot(ax=ax[0], grid=True, title="Train", color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test", color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()
        
    ## split exog
    if exog is not None:
        exog_train = exog[0:split] 
        exog_test = exog[split:]
        return ts_train, ts_test, exog_train, exog_test
    else:
        return ts_train, ts_test



'''
Compute the confidence interval for predictions:
    [y[t+h] +- (c*σ*√h)]
:parameter
    :param lst_values: list or array
    :param error_std: σ (standard dev of residuals)
    :param conf: num - confidence level (90%, 95%, 99%)
:return
    array with 2 columns (upper and lower bounds)
'''
def utils_conf_int(lst_values, error_std, conf=0.95):
    lst_values = list(lst_values) if type(lst_values) != list else lst_values
    c = round( stats.norm.ppf(1-(1-conf)/2), 2)
    lst_ci = []
    for x in lst_values:
        lst_x = lst_values[:lst_values.index(x)+1]
        h = len(lst_x)
        ci = [x - (c*error_std*np.sqrt(h)), x + (c*error_std*np.sqrt(h))]
        lst_ci.append(ci)
    return np.array(lst_ci)



'''
Evaluation metrics for predictions.
:parameter
    :param dtf: DataFrame with columns "ts", "model", "forecast", and "lower"/"upper" (if available)
:return
    dtf with columns "ts", "model", "residuals", "lower", "forecast", "upper", "error"
'''
def utils_evaluate_ts_model(dtf, conf=0.95, title=None, plot=True, figsize=(20,13)):
    try:
        ## residuals from fitting
        ### add column
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        ### kpi
        residuals_mean = dtf["residuals"].mean()
        residuals_std = dtf["residuals"].std()

        ## forecasting error
        ### add column
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        ### kpi
        error_mean = dtf["error"].mean() 
        error_std = dtf["error"].std() 
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["error"].apply(lambda x: x**2).mean()  #mean squared error
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## interval
        if "upper" not in dtf.columns:
            print("--- computing confidence interval ---")
            dtf["lower"], dtf["upper"] = [np.nan, np.nan]
            dtf.loc[dtf["forecast"].notnull(), ["lower","upper"]] = utils_conf_int(
                dtf[dtf["forecast"].notnull()]["forecast"], residuals_std, conf)
        
        ## plot
        if plot is True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Train (obs: "+str(len(dtf[pd.notnull(dtf["model"])]))+")", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Test (obs: "+str(len(dtf[pd.isnull(dtf["model"])]))+")", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['lower'], y2=dtf['upper'], color='b', alpha=0.2)
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.axvline(dtf["residuals"].mean(), ls='--', color="green", label="mean: "+str(round(dtf["residuals"].mean(),2)))
            ax4.axvline(dtf["error"].mean(), ls='--', color="red", label="mean: "+str(round(dtf["error"].mean(),2)))
            ax4.set(ylabel=None)
            ax4.legend()
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[["ts", "model", "residuals", "lower", "forecast", "upper", "error"]]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
    


'''
Generate dates to index predictions.
:parameter
    :param start: str - "yyyy-mm-dd"
    :param end: str - "yyyy-mm-dd"
    :param n: num - length of index
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
'''
def utils_generate_indexdate(start, end=None, n=None, freq="D"):
    if end is not None:
        index = pd.date_range(start=start, end=end, freq=freq)
    else:
        index = pd.date_range(start=start, periods=n, freq=freq)
    index = index[1:]
    print("--- generating index date --> start:", index[0], "| end:", index[-1], "| len:", len(index), "---")
    return index



'''
Plot unknown future forecast and produce conf_int with residual_std and pred_int if an error_std is given.
:parameter
    :param dtf: DataFrame with columns "ts", "model", "forecast", and "lower"/"upper" (if available)
    :param conf: num - confidence level (90%, 95%, 99%)
    :param zoom: int - plots the focus on the last zoom days
:return
    dtf with columns "ts", "model", "residuals", "lower", "forecast", "upper" (No error)
'''
def utils_add_forecast_int(dtf, conf=0.95, plot=True, zoom=30, figsize=(15,5)):
    ## residuals from fitting
    ### add column
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    ### kpi
    residuals_std = dtf["residuals"].std()
    
    ## interval
    if "upper" not in dtf.columns:
        print("--- computing confidence interval ---")
        dtf["lower"], dtf["upper"] = [np.nan, np.nan]
        dtf.loc[dtf["forecast"].notnull(), ["lower","upper"]] = utils_conf_int(
            dtf[dtf["forecast"].notnull()]["forecast"], residuals_std, conf)

    ## plot
    if plot is True:
        fig = plt.figure(figsize=figsize)
        
        ### entire series
        ax0 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=2)
        dtf[["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax0, title="History + Future")
        ax0.fill_between(x=dtf.index, y1=dtf['lower'], y2=dtf['upper'], color='b', alpha=0.2)
        ax0.set(xlabel=None)

        ### focus on last
        ax1 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
        first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
        first_loc = dtf.index.tolist().index(first_idx)
        zoom_idx = dtf.index[first_loc-zoom]
        dtf.loc[zoom_idx:][["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax1, title="Zoom on the last "+str(zoom)+" observations")
        ax1.fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['lower'], y2=dtf.loc[zoom_idx:]['upper'], color='b', alpha=0.2)
        ax1.set(xlabel=None)
        plt.show()
    return dtf[["ts", "model", "residuals", "lower", "forecast", "upper"]]

        
###############################################################################
#                        AUTOREGRESSIVE                                       #
###############################################################################
'''
Tune Holt-Winters Exponential Smoothing
:parameter
    :param ts_train: pandas timeseries
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param val_size: num - size of validation fold
    :param scoring: function(y_true, y_pred)
    :param top: num - plot top models only
:return
    dtf with results
'''
def tune_expsmooth_model(ts_train, s=7, val_size=0.2, scoring=None, top=None, figsize=(15,5)):
    ## split
    dtf_fit, dtf_val = model_selection.train_test_split(ts_train, test_size=val_size, shuffle=False)
    dtf_fit, dtf_val = dtf_fit.to_frame(name="ts"), dtf_val.to_frame(name="ts")
    
    ## scoring
    scoring = metrics.mean_absolute_error if scoring is None else scoring   
    
    ## hyperamater space
    trend = ['add', 'mul', None]
    damped = [True, False]
    seasonal = ['add', 'mult', None]

    ## grid search
    dtf_search = pd.DataFrame(columns=["combo","score","model"])
    combinations = []
    for t in trend:
        for d in damped:
            for ss in seasonal:
                combo = "trend="+str(t)+", damped="+str(d)+", seas="+str(ss)
                if combo not in combinations:
                    combinations.append(combo)
                    try:
                        ### fit
                        model = smt.ExponentialSmoothing(dtf_fit, trend=t, damped=d, seasonal=ss, seasonal_periods=s).fit()
                        ### predict
                        pred =  model.forecast(len(dtf_val))
                        if pred.isna().sum() == 0:
                            dtf_val[combo] = pred.values
                            score = scoring(dtf_val["ts"].values, dtf_val[combo].values)
                            dtf_search = dtf_search.append(pd.DataFrame({"combo":[combo],"score":[score],"model":[model]}))
                    except:
                        continue
    
    ## find best
    dtf_search = dtf_search.sort_values("score").reset_index(drop=True)
    best = dtf_search["combo"].iloc[0]
    dtf_val = dtf_val.rename(columns={best:best+" [BEST]"})
    dtf_val = dtf_val[["ts",best+" [BEST]"] + list(dtf_search["combo"].unique())[1:]]
    
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("Model Tuning", fontsize=15)
    combos = dtf_val.drop("ts", axis=1).columns[:top]
    if (len(combos) <= 7) or ((top is not None) and (top <= 7)):
        colors = ["red","blue","green","violet","sienna","orange","yellow"]
    else: 
        colors = [tuple(np.random.rand(3,)) for i in range(len(combos))]

    ### main
    ts_train.plot(ax=ax[0], grid=True, color="black", legend=True, label="ts")
    ax[0].fill_between(x=dtf_fit.index, y1=ts_train.max(), color='grey', alpha=0.2)
    dtf_val[combos].plot(grid=True, ax=ax[0], color=colors, legend=True)
    ax[0].legend(loc="upper left")
    ax[0].set(xlabel=None)
    ### zoom
    dtf_val["ts"].plot(grid=True, ax=ax[1], color="black", legend=False) 
    for i,col in enumerate(combos):
        linewidth = 2 if col == best+" [BEST]" else 1
        dtf_val[col].plot(grid=True, ax=ax[1], color=colors[i], legend=False, linewidth=linewidth)
    ax[1].set(xlabel=None)  
    plt.show()
    return dtf_search



'''
Fits Exponential Smoothing: 
    Simple (level) --> trend=None + seasonal=None
        y[t+i] = α*y[t] + α(1-α)^1*y[t-1] + α(1-α)^2*y[t-2] + ... = (α)*y[t] + (1-α)*yhat[t]
    Holt (level + trend) --> trend=["add","mul"] + seasonal=None
        y[t+i] = level_f(α) + i*trend_f(β)
    Holt-Winters (level + trend + seasonality) --> trend=["add","mul"] + seasonal=["add","mul"]
        y[t+i] = level_f(α) + i*trend_f(β) + seasonality_f(γ)
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param trend: str - "additive" (linear), "multiplicative" (non-linear)
    :param damped: bool - damp trend
    :param seasonal: str - "additive" (ex. +100 every 7 days), "multiplicative" (ex. x10 every 7 days)
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param factors: tuple - (α,β,γ) smoothing factor for the level (ex 0.94), trend, seasonal
:return
    dtf with predictons and the model
'''
def fit_expsmooth(ts_train, ts_test, trend="additive", damped=False, seasonal="multiplicative", s=None, factors=(None,None,None), conf=0.95, figsize=(15,10)):
    ## checks
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal is None) & (s is None) else "Seasonal parameters: "+str(seasonal)+" Seasonality every "+str(s)+" observations"
    print(check_seasonality)
    
    ## train
    model = smt.ExponentialSmoothing(ts_train, trend=trend, damped=damped, seasonal=seasonal, seasonal_periods=s).fit(factors[0], factors[1], factors[2])
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    alpha, beta, gamma = round(model.params["smoothing_level"],2), round(model.params["smoothing_slope"],2), round(model.params["smoothing_seasonal"],2)
    dtf = utils_evaluate_ts_model(dtf, conf=conf, figsize=figsize, title="Holt-Winters "+str((alpha, beta, gamma)))
    return dtf, model

'''
Tune ARIMA
:parameter
    :param ts_train: pandas timeseries
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param val_size: num - size of validation fold
    :param max_order: tuple - max (p,d,q) values
    :param seasonal_order: tuple - max (P,D,Q) values
    :param scoring: function(y_true, y_pred)
    :param top: num - plot top models only
:return
    dtf with results
'''
def tune_arima_model(ts_train, s=7, val_size=0.2, max_order=(3,1,3), seasonal_order=(1,1,1), scoring=None, top=None, figsize=(15,5)):
    ## split
    dtf_fit, dtf_val = model_selection.train_test_split(ts_train, test_size=val_size, shuffle=False)
    dtf_fit, dtf_val = dtf_fit.to_frame(name="ts"), dtf_val.to_frame(name="ts")
    
    ## scoring
    scoring = metrics.mean_absolute_error if scoring is None else scoring   
    
    ## hyperamater space
    ps = range(0,max_order[0]+1)
    ds = range(0,max_order[1]+1)
    qs = range(0,max_order[2]+1)
    Ps = range(0,seasonal_order[0]+1)
    Ds = range(0,seasonal_order[1]+1)
    Qs = range(0,seasonal_order[2]+1)

    ## grid search
    dtf_search = pd.DataFrame(columns=["combo","score","model"])
    combinations = []
    for p in ps:
        for d in ds:
            for q in qs:
                for P in Ps:
                    for D in Ds:
                        for Q in Qs:
                            combo = "("+str(p)+","+str(d)+","+str(q)+")x("+str(P)+","+str(D)+","+str(Q)+")"
                            if combo not in combinations:
                                combinations.append(combo)
                            try:
                                ### fit
                                model = smt.SARIMAX(ts_train, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
                                ### predict
                                pred =  model.forecast(len(dtf_val))
                                if pred.isna().sum() == 0:
                                    dtf_val[combo] = pred.values
                                    score = scoring(dtf_val["ts"].values, dtf_val[combo].values)
                                    dtf_search = dtf_search.append(pd.DataFrame({"combo":[combo],"score":[score],"model":[model]}))
                            except:
                                continue
    
    ## find best
    dtf_search = dtf_search.sort_values("score").reset_index(drop=True)
    best = dtf_search["combo"].iloc[0]
    dtf_val = dtf_val.rename(columns={best:best+" [BEST]"})
    dtf_val = dtf_val[["ts",best+" [BEST]"] + list(dtf_search["combo"].unique())[1:]]
    
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("Model Tuning", fontsize=15)
    combos = dtf_val.drop("ts", axis=1).columns[:top]
    if (len(combos) <= 7) or ((top is not None) and (top <= 7)):
        colors = ["red","blue","green","violet","sienna","orange","yellow"]
    else: 
        colors = [tuple(np.random.rand(3,)) for i in range(len(combos))]

    ### main
    ts_train.plot(ax=ax[0], grid=True, color="black", legend=True, label="ts")
    ax[0].fill_between(x=dtf_fit.index, y1=ts_train.max(), color='grey', alpha=0.2)
    dtf_val[combos].plot(grid=True, ax=ax[0], color=colors, legend=True)
    ax[0].legend(loc="upper left")
    ax[0].set(xlabel=None)
    ### zoom
    dtf_val["ts"].plot(grid=True, ax=ax[1], color="black", legend=False) 
    for i,col in enumerate(combos):
        linewidth = 2 if col == best+" [BEST]" else 1
        dtf_val[col].plot(grid=True, ax=ax[1], color=colors[i], legend=False, linewidth=linewidth)
    ax[1].set(xlabel=None)  
    plt.show()
    return dtf_search


    
'''
Find best Seasonal-ARIMAX parameters.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    best model
'''
def find_best_sarimax(ts, seasonal=True, stationary=False, s=1, exog=None,
                      max_p=10, max_d=3, max_q=10,
                      max_P=10, max_D=3, max_Q=10):
    best_model = pmdarima.auto_arima(ts, exogenous=exog,
                                     seasonal=seasonal, stationary=stationary, m=s, 
                                     information_criterion='aic', max_order=20,
                                     max_p=max_p, max_d=max_d, max_q=max_q,
                                     max_P=max_P, max_D=max_D, max_Q=max_Q,
                                     error_action='ignore')
    print("best model --> (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
    return best_model.summary()



'''
Fits SARIMAX (Seasonal ARIMA with External Regressors) (p,d,q)x(P,D,Q,s):  
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] + b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param order: tuple - (p,d,q) --> p: lag order (AR), d: degree of differencing (to remove trend), q: order of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q) --> seasonal lag orders (ex. lag from the last 2 seasons)
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param exog_train: pandas dataframe or numpy array
    :param exog_test: pandas dataframe or numpy array
:return
    dtf with predictons and the model
'''
def fit_sarimax(ts_train, ts_test, order=(1,0,1), seasonal_order=(1,0,1), s=7, exog_train=None, exog_test=None, conf=0.95, figsize=(15,10)):
    ## checks
    check_trend = "Trend parameters: No differencing" if order[1] == 0 else "Trend parameters: d="+str(order[1])
    print(check_trend)
    check_seasonality = "Seasonal parameters: No Seasonality" if (s == 0) & (np.sum(seasonal_order[0:2]) == 0) else "Seasonal parameters: Seasonality every "+str(s)+" observations"
    print(check_seasonality)
    check_exog = "Exog parameters: Not given" if (exog_train is None) & (exog_test is None) else "Exog parameters: number of regressors="+str(exog_train.shape[1])
    print(check_exog)
    
    ## train
    model = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order+(s,), exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit()
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1, exog=exog_test)

    ## add conf_int
    ci = model.get_forecast(len(ts_test)).conf_int(1-conf).values
    dtf_test["lower"], dtf_test["upper"] = ci[:,0], ci[:,1]
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_ts_model(dtf, conf=conf, figsize=figsize, title=title)
    return dtf, model


'''
Forecast unknown future with sarimax or expsmooth.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_autoregressive(ts, model=None, pred_ahead=None, end=None, freq="D", conf=0.95, zoom=30, figsize=(15,5)):
    ## model
    model = smt.SARIMAX(ts, order=(1,1,1), seasonal_order=(0,0,0,0)).fit() if model is None else model 

    ## fit
    dtf = ts.to_frame(name="ts")
    dtf["model"] = model.fittedvalues
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    if "holtwinters" in str(model):
        preds = model.forecast(len(index))
        dtf_preds = preds.to_frame(name="forecast")
    else:
        preds = model.get_forecast(len(index))
        dtf_preds = preds.predicted_mean.to_frame(name="forecast")
        ci = preds.conf_int(1-conf).values
        dtf_preds["lower"], dtf_preds["upper"] = ci[:,0], ci[:,1]
        
    ## add intervals and plot
    dtf = dtf.append(dtf_preds)
    dtf = utils_add_forecast_int(dtf, conf=conf, zoom=zoom)
    return dtf



###############################################################################
#                            RNN                                              #
###############################################################################
    
    
    
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()
    
    
    
'''
Preprocess a ts for LSTM partitioning into X and y.
:parameter
    :param ts: pandas timeseries
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param scaler: sklearn scaler object - if None is fitted
    :param exog: pandas dataframe or numpy array
:return
    X with shape: (len(ts)-s, s, features)
    y with shape: (len(ts)-s,)
    the fitted scaler
'''
def utils_preprocess_lstm(ts, s, scaler=None, exog=None):
    ## scale
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ts_preprocessed = scaler.fit_transform(ts.values.reshape(-1,1)).reshape(-1)        
    
    ## create X (N,s,x)  and y (N,)
    ts_preprocessed = kprocessing.sequence.TimeseriesGenerator(data=ts_preprocessed, 
                                                               targets=ts_preprocessed, 
                                                               length=s, batch_size=1)
    lst_X, lst_y = [], []
    for i in range(len(ts_preprocessed)):
        xi, yi = ts_preprocessed[i]
        lst_X.append(xi[0])
        lst_y.append(yi[0])
    X = np.expand_dims(np.array(lst_X), axis=2)
    y = np.array(lst_y)
    return X, y, scaler



'''
Get fitted values from LSTM.
'''
def utils_fitted_lstm(ts, model, scaler, exog=None):
    ## scale
    s = model.input_shape[1]
    ts_preprocessed = scaler.transform(ts.values.reshape(-1,1)).reshape(-1) 
    
    ## create Xy, predict = fitted
    lst_fitted = [np.nan]*s
    for i in range(len(ts_preprocessed)):
        end_ix = i + s
        if end_ix > len(ts_preprocessed)-1:
            break
        X = ts_preprocessed[i:end_ix]
        X = np.array(X)
        X = np.reshape(X, (1,s,1))
        fit = model.predict(X)
        fit = scaler.inverse_transform(fit)[0][0]
        lst_fitted.append(fit)
    return np.array(lst_fitted)



'''
Predict ts with LSTM using previous predictions.
'''
def utils_predict_lstm(last_s_obs, model, scaler, pred_ahead, exog=None):
    ## scale
    s = model.input_shape[1]
    ts_preprocessed = list(scaler.transform(last_s_obs.values.reshape(-1,1))) 
    
    ## predict, append, re-predict
    lst_preds = []
    for i in range(pred_ahead):
        X = np.array(ts_preprocessed[len(ts_preprocessed)-s:])
        X = np.reshape(X, (1,s,1))
        pred = model.predict(X)
        ts_preprocessed.append(pred[0])
        pred = scaler.inverse_transform(pred)[0][0]
        lst_preds.append(pred)
    return np.array(lst_preds)



'''
Fit Long Short-Term Memory neural network.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    dtf with predictons and the model 
'''
def fit_lstm(ts_train, ts_test, model, exog=None, s=20, epochs=100, conf=0.95, figsize=(15,5)):
    ## check
    print("Seasonality: using the last", s, "observations to predict the next 1")
    
    ## preprocess train
    X_train, y_train, scaler = utils_preprocess_lstm(ts_train, scaler=None, exog=exog, s=s)
    print("--- X:", X_train.shape, "| y:", y_train.shape, "---")
    
    ## lstm
    if model is None:
        model = models.Sequential()
        model.add( layers.LSTM(input_shape=X_train.shape[1:], units=50, activation='relu', return_sequences=False) )
        model.add( layers.Dense(1) )
        model.compile(optimizer='adam', loss='mean_absolute_error')
        print(model.summary())
        
    ## train
    verbose = 0 if epochs > 1 else 1
    training = model.fit(x=X_train, y=y_train, batch_size=1, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3)
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = utils_fitted_lstm(ts_train, training.model, scaler, exog)
    dtf_train["model"] = dtf_train["model"].fillna(method='bfill')
    
    ## test
    last_s_obs = ts_train[-s:]
    preds = utils_predict_lstm(last_s_obs, training.model, scaler, pred_ahead=len(ts_test), exog=None)
    dtf_test = ts_test.to_frame(name="ts").merge(pd.DataFrame(data=preds, index=ts_test.index, columns=["forecast"]),
                                                 how='left', left_index=True, right_index=True)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_ts_model(dtf, conf=conf, figsize=figsize, title="LSTM (memory:"+str(s)+")")
    return dtf, training.model



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_lstm(ts, model=None, epochs=100, pred_ahead=None, end=None, freq="D", conf=0.95, zoom=30, figsize=(15,5)):
    ## model
    if model is None:
        model = models.Sequential([
            layers.LSTM(input_shape=(1,1), units=50, activation='relu', return_sequences=False),
            layers.Dense(1) ])
        model.compile(optimizer='adam', loss='mean_absolute_error')

    ## fit
    s = model.input_shape[1]
    X, y, scaler = utils_preprocess_lstm(ts, scaler=None, exog=None, s=s)
    training = model.fit(x=X, y=y, batch_size=1, epochs=epochs, shuffle=True, verbose=0, validation_split=0.3)
    dtf = ts.to_frame(name="ts")
    dtf["model"] = utils_fitted_lstm(ts, training.model, scaler, None)
    dtf["model"] = dtf["model"].fillna(method='bfill')
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    last_s_obs = ts[-s:]
    preds = utils_predict_lstm(last_s_obs, training.model, scaler, pred_ahead=len(index), exog=None)
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## add intervals and plot
    dtf = utils_add_forecast_int(dtf, conf=conf, zoom=zoom)
    return dtf


###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet on Business Data:
    y = trend + seasonality + holidays
:parameter
    :param dtf_train: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param dtf_test: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param lst_exog: list - names of variables
    :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS" monthly start ...
:return
    dtf with predictons and the model
'''
def fit_prophet(dtf_train, dtf_test, lst_exog=None, model=None, freq="D", conf=0.95, figsize=(15,10)):
    ## setup prophet
    if model is None:
        model = Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                        yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                        holidays=None, interval_width=conf)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)
    
    ## train
    model.fit(dtf_train)
    
    ## test
    dtf_prophet = model.make_future_dataframe(periods=len(dtf_test), freq=freq, include_history=True)
    
    if model.growth == "logistic":
        dtf_prophet["cap"] = dtf_train["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(dtf_train[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-len(dtf_test):][lst_exog] = dtf_test[lst_exog].values
    
    dtf_prophet = model.predict(dtf_prophet)
    dtf_train = dtf_train.merge(dtf_prophet[["ds","yhat"]], how="left").rename(
        columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    dtf_test = dtf_test.merge(dtf_prophet[["ds","yhat","yhat_lower","yhat_upper"]], how="left").rename(
        columns={'yhat':'forecast', 'y':'ts', 'yhat_lower':'lower', 'yhat_upper':'upper'}).set_index("ds")
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_ts_model(dtf, conf=conf, figsize=figsize, title="Prophet")
    return dtf, model
    


'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_prophet(dtf, model=None, pred_ahead=None, end=None, freq="D", conf=0.95, zoom=30, figsize=(15,5)):
    ## model
    model = Prophet() if model is None else model

    ## fit
    model.fit(dtf)
    
    ## index
    index = utils_generate_indexdate(start=dtf["ds"].values[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    dtf_prophet = model.make_future_dataframe(periods=len(index), freq=freq, include_history=True)
    dtf_prophet = model.predict(dtf_prophet)
    dtf = dtf.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    preds = pd.DataFrame(data=index, columns=["ds"])
    preds = preds.merge(dtf_prophet[["ds","yhat","yhat_lower","yhat_upper"]], how="left").rename(
        columns={'yhat':'forecast', 'yhat_lower':'lower', 'yhat_upper':'upper'}).set_index("ds")
    dtf = dtf.append(preds)
    
    ## plot
    dtf = utils_add_forecast_int(dtf, conf=conf, zoom=zoom)
    return dtf
