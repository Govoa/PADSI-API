from flask import Flask
from flask_cors import CORS
import ccxt
from ccxt.base.errors import RequestTimeout
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
import numpy as np
import pandas as pd
from joblib import dump, load
from tensorflow import keras

binance = ccxt.binance()
# Cargamos los modelos de KNN
knn25 = load('./Classifiers/KNN/KNN25S.joblib')
knn15 = load('./Classifiers/KNN/KNN15S.joblib')
knn5 = load('./Classifiers/KNN/KNN5S.joblib')
# Cargamos los modelos de SVM
svm25s = load('./Classifiers/SVM/SVM25S.joblib')
svm15s = load('./Classifiers/SVM/SVM15S.joblib')
svm5s = load('./Classifiers/SVM/SVM5S.joblib')

# Cargamos los modelos de DNN
dnn25s = keras.models.load_model('./Classifiers/DNN/DNN25S.h5')
dnn15s = keras.models.load_model('./Classifiers/DNN/DNN15S.h5')
dnn5s = keras.models.load_model('./Classifiers/DNN/DNN5S.h5')

app = Flask(__name__)
CORS(app)

def getTrades(symbol,timefreq):

    endDate = datetime.now()
    startDate = endDate - timedelta(seconds=timefreq)

    start = int(startDate.timestamp() * 1000) #miliseconds
    end = int(endDate.timestamp() * 1000) #miliseconds

    records = []
    since = start
    five_seconds = 5000 

    while since < end:
        try:
            orders = binance.fetch_trades(symbol + '/BTC', since)
        except RequestTimeout:
            time.sleep(5)
            orders = binance.fetch_trades(symbol + '/BTC', since)

        if len(orders) > 0:

            latest_ts = orders[-1]['timestamp']
            if since != latest_ts:
                since = latest_ts
            else:
                since += five_seconds

            for l in orders:
                records.append({
                    'symbol': l['symbol'],
                    'timestamp': l['timestamp'],
                    'datetime': l['datetime'],
                    'side': l['side'],
                    'price': l['price'],
                    'amount': l['amount'],
                    'btc_volume': float(l['price']) * float(l['amount']),
                })
        else:
            since += five_seconds

    return pd.DataFrame.from_records(records)

def std_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def avg_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    return results

def std_trades_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].count()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def std_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def avg_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    return results

def std_price_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results

def avg_price_feature(df_buy_rolling):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results

def avg_price_max(df_buy_rolling):
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results

def chunks_time(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    #the index contains time info
    return buy_volume.index

def build_features(trades, coin, time_freq, rolling_freq, index):
    df = trades
    df["time"] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    df = df.reset_index().set_index('time')

    df_buy = df[df['side'] == 'buy']

    df_buy_grouped = df_buy.groupby(pd.Grouper(freq=time_freq))

    date = chunks_time(df_buy_grouped)

    results_df = pd.DataFrame(
        {'date': date,
         'pump_index': index,
         'std_rush_order': std_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         'avg_rush_order': avg_rush_order_feature(df_buy, time_freq, rolling_freq).values,
         'std_trades': std_trades_feature(df_buy_grouped, rolling_freq).values,
         'std_volume': std_volume_feature(df_buy_grouped, rolling_freq).values,
         'avg_volume': avg_volume_feature(df_buy_grouped, rolling_freq).values,
         'std_price': std_price_feature(df_buy_grouped, rolling_freq).values,
         'avg_price': avg_price_feature(df_buy_grouped),
         'avg_price_max': avg_price_max(df_buy_grouped).values,
         'hour_sin': np.sin(2 * np.pi * date.hour/23),
         'hour_cos': np.cos(2 * np.pi * date.hour/23),
         'minute_sin': np.sin(2 * np.pi * date.minute / 59),
         'minute_cos': np.cos(2 * np.pi * date.minute / 59),
         })

    results_df['symbol'] = coin
    results_df['gt'] = 0
    results_df = results_df.dropna()
    return results_df

def predictKNN25(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = knn25.predict(input)
    return output

def predictKNN15(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = knn15.predict(input)
    return output

def predictKNN5(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = knn5.predict(input)
    return output

def predictSVM25(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = svm25s.predict(input)
    return output

def predictSVM15(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = svm15s.predict(input)
    return output

def predictSVM5(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = svm5s.predict(input)
    return output

def predictDNN25(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = dnn25s.predict(input)
    return output

def predictDNN15(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = dnn15s.predict(input)
    return output

def predictDNN5(input):
    features = ['std_rush_order',
            'avg_rush_order',
            'std_trades',
            'std_volume',
            'avg_volume',
            'std_price',
            'avg_price',
            'avg_price_max',
            'hour_sin',
            'hour_cos',
            'minute_sin',
            'minute_cos']
    input = input[features].iloc[-1]
    input = np.array(input).reshape(1,-1)
    output = dnn5s.predict(input)
    return output


@app.route('/getKNN25/<token>')
def getKNN25(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'25s', 900, 0)
    output = predictKNN25(inputs)
    return str(output)

@app.route('/getKNN15/<token>')
def getKNN15(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'15s', 900, 0)
    output = predictKNN15(inputs)
    return str(output)

@app.route('/getKNN5/<token>')
def getKNN5(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'5s', 900, 0)
    output = predictKNN5(inputs)
    return str(output)

@app.route('/getSVM25/<token>')
def getSVM25(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'25s', 900, 0)
    output = predictSVM25(inputs)
    return str(output)

@app.route('/getSVM15/<token>')
def getSVM15(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'15s', 900, 0)
    output = predictSVM15(inputs)
    return str(output)

@app.route('/getSVM5/<token>')
def getSVM5(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'5s', 900, 0)
    output = predictSVM5(inputs)
    return str(output)

@app.route('/getDNN25/<token>')
def getDNN25(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'25s', 900, 0)
    output = predictDNN25(inputs)
    return str(output)

@app.route('/getDNN15/<token>')
def getDNN15(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'15s', 900, 0)
    output = predictDNN15(inputs)
    return str(output)

@app.route('/getDNN5/<token>')
def getDNN5(token):
    trades = getTrades(token,35000)
    inputs = build_features(trades,token,'5s', 900, 0)
    output = predictDNN5(inputs)
    return str(output)


if __name__ == "__main__":
    app.run(debug=True)