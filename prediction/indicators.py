import itertools

import pandas as pd
import numpy as np

from log import Log


def gen_EMA(data: pd.Series, n=20):
    alpha = 1 / (n + 1)
    EMA = []
    for t in range(len(data.index)):
        if t == 0:
            EMA_t = data.iat[t]
        else:
            EMA_t = alpha * data.iat[t] + (1 - alpha) * EMA[-1]
        EMA.append(EMA_t)
    return EMA


def gen_RSI(data: pd.Series, n=14):
    RSI = [None] * n
    for t in range(n, len(data.index)):
        sum_up = 0
        sum_down = 0
        for i in range(1, n):
            sum_up += max(data.iat[t - i + 1] - data.iat[t - i], 0)
            sum_down += min(data.iat[t - i + 1] - data.iat[t - i], 0)
        avg_up = sum_up / n
        avg_down = -sum_down / n
        RSI_t = avg_up / (avg_up + avg_down)
        RSI.append(RSI_t)
    return pd.Series(RSI, index=data.index)


def gen_Stochastics(data: pd.Series, K_n=5, D=5, D2=3):
    K = [0] * K_n
    D_fast = [None] * K_n
    D_slow = [None] * K_n
    for t in range(K_n, len(data.index)):
        L = min(filter(None, data[(t + 1 - K_n):(t + 1)]))
        H = max(filter(None, data[(t + 1 - K_n):(t + 1)]))
        K_t = 100 * (data.iat[t] - L) / (H - L)
        K.append(K_t)
        D_fast_t = sum(filter(None, K[-D:])) / D
        D_fast.append(D_fast_t)
        D_slow_t = sum(filter(None, D_fast[-D2:])) / D2
        D_slow.append(D_slow_t)
    return pd.Series(K, index=data.index), \
           pd.Series(D_fast, index=data.index), \
           pd.Series(D_slow, index=data.index)


def gen_Williams_R(data: pd.Series, n=10):
    Williams_R = [None] * n
    for t in range(n, len(data.index)):
        L = min(filter(None, data[(t + 1 - n):(t + 1)]))
        H = max(filter(None, data[(t + 1 - n):(t + 1)]))
        R_t = (H - data.iat[t]) / (H - L) * -100
        Williams_R.append(R_t)
    return pd.Series(Williams_R, index=data.index)


def gen_price_rate_of_change(data: pd.Series, n=10):
    proc = [None] * n
    for t in range(n, len(data.index)):
        proc_t = (data.iat[t] - data.iat[t - n]) / data.iat[t - n]
        proc.append(proc_t)
    return pd.Series(proc, index=data.index)


def gen_on_balance_volume(data: pd.DataFrame):
    obv = [0]
    for t in range(1, len(data.index)):
        if data["Close"][t] > data["Close"][t - 1]:
            obv_t = obv[-1] + data["Volume"][t]
        elif data["Close"][t] < data["Close"][t - 1]:
            obv_t = obv[-1] - data["Volume"][t]
        else:
            obv_t = obv[-1]
        obv.append(obv_t)
    return pd.Series(obv, index=data.index)


def gen_MACD(data: pd.Series, n_fast=12, n_slow=26, n_signal=9):
    alpha_fast = 1 / (n_fast + 1)
    alpha_slow = 1 / (n_slow + 1)
    alpha_signal = 1 / (n_signal + 1)

    EMA_fast = []
    EMA_slow = []
    MACD = []
    Signal = []

    for t in range(len(data.index)):
        if t == 0:
            EMA_fast_t = data.iat[t]
            EMA_slow_t = data.iat[t]
            MACD_t = EMA_fast_t - EMA_slow_t
            Signal_t = MACD_t
        else:
            EMA_fast_t = alpha_fast * data.iat[t] + (1 - alpha_fast) * EMA_fast[-1]
            EMA_slow_t = alpha_slow * data.iat[t] + (1 - alpha_slow) * EMA_slow[-1]
            MACD_t = EMA_fast_t - EMA_slow_t
            Signal_t = alpha_signal * MACD_t + (1 - alpha_signal) * Signal[-1]
        EMA_fast.append(EMA_fast_t)
        EMA_slow.append(EMA_slow_t)
        MACD.append(MACD_t)
        Signal.append(Signal_t)
    return pd.Series(MACD, index=data.index), \
           pd.Series(Signal, index=data.index)


def gen_CCI(data: pd.DataFrame, n=20):
    CCI = [None] * n
    TP = []

    for t in range(len(data.index)):
        TP_t = (data["High"].iat[t] + data["Low"].iat[t] + data["Close"].iat[t]) / 3
        TP.append(TP_t)
        if t >= n:
            SMA = np.mean(TP[-n:])
            MD = sum(abs(TP[-n:] - SMA)) / n
            CCI_t = (TP_t - SMA) / (0.015 * MD)
            CCI.append(CCI_t)
    return pd.Series(CCI, index=data.index)


def gen_ATR(data: pd.DataFrame, n=14):
    TR = []
    ATR = [None] * n

    for t in range(len(data.index)):
        if t == 0:
            continue
        TR_t = max(data["High"].iat[t] - data["Low"].iat[t],
                   abs(data["High"].iat[t] - data["Close"].iat[t - 1]),
                   abs(data["Low"].iat[t] - data["Close"].iat[t - 1]))
        TR.append(TR_t)
        if t == n:
            ATR_t = np.mean(TR)
            ATR.append(ATR_t)
        elif t > n:
            ATR_t = (ATR[-1] * (n - 1) + TR_t) / n
            ATR.append(ATR_t)
    return pd.Series(ATR, index=data.index)


def gen_ADL(data: pd.DataFrame):
    ADL = []
    for t in range(len(data.index)):
        ADL_t = ((data["Close"].iat[t] - data["Low"].iat[t]) - \
                 (data["High"].iat[t] - data["Close"].iat[t])) / \
                (data["High"].iat[t] - data["Low"].iat[t]) * data["Volume"].iat[t]
        if t == 0:
            ADL.append(ADL_t)
        else:
            ADL.append(ADL[-1] + ADL_t)
    return pd.Series(ADL, index=data.index)


def gen_returns(data: pd.DataFrame):
    returns = [None]
    log_returns = [None]
    annualized_log_returns = [None]
    monthly_log_returns = [None] * 21
    quarterly_log_returns = [None] * 63
    yearly_log_returns = [None] * 252

    for t in range(1, len(data.index)):
        return_t = (data["Close"].iat[t] - data["Close"].iat[t - 1]) / data["Close"].iat[t - 1]
        log_return_t = np.log(data["Close"].iat[t] / data["Close"].iat[t - 1])
        if t >= 21:
            monthly_log_return_t = np.log(data["Close"].iat[t] / data["Close"].iat[t - 21])
            monthly_log_returns.append(monthly_log_return_t)
        if t >= 63:
            quarterly_log_return_t = np.log(data["Close"].iat[t] / data["Close"].iat[t - 63])
            quarterly_log_returns.append(quarterly_log_return_t)
        if t >= 252:
            yearly_log_return_t = np.log(data["Close"].iat[t] / data["Close"].iat[t - 252])
            yearly_log_returns.append(yearly_log_return_t)

        returns.append(return_t)
        log_returns.append(log_return_t)
        annualized_log_returns.append(252 * log_return_t)

    return pd.Series(returns, index=data.index), \
           pd.Series(log_returns, index=data.index), \
           pd.Series(annualized_log_returns, index=data.index), \
           pd.Series(monthly_log_returns, index=data.index), \
           pd.Series(quarterly_log_returns, index=data.index), \
           pd.Series(yearly_log_returns, index=data.index)


def gen_returns2(data: pd.DataFrame, delta=1):
    returns = []
    log_returns = []
    ann_log_returns = []

    for t in range(len(data.index) - delta):
        return_t = (data["Close"].iat[t + delta] - data["Close"].iat[t]) / data["Close"].iat[t]
        log_return_t = np.log(data["Close"].iat[t + delta] / data["Close"].iat[t])
        ann_log_returns_t = log_return_t / delta * 252

        returns.append(return_t)
        log_returns.append(log_return_t)
        ann_log_returns.append(ann_log_returns_t)

    for i in range(delta):
        returns.append(None)
        log_returns.append(None)
        ann_log_returns.append(None)

    return pd.Series(returns, index=data.index), \
           pd.Series(log_returns, index=data.index), \
           pd.Series(ann_log_returns, index=data.index)


def gen_SimpleVola(data: pd.Series, days=14):
    days_year = 252
    vola = [None] * days
    ann_vola = [None] * days
    log_returns = [None]
    ann_log_returns = [None]

    for t in range(1, len(data.index)):
        log_return_t = np.log(data.iat[t] / data.iat[t - 1])
        log_returns.append(log_return_t)
        ann_log_returns.append(log_return_t * days_year)

        if t >= days:
            vola.append(np.std(log_returns[-days:]))
            ann_vola.append(np.std(ann_log_returns[-days:]))

    return pd.Series(vola, index=data.index), \
           pd.Series(ann_vola, index=data.index)


def gen_EWMA_Vola(data: pd.Series, n=14):
    lambda_ = 0.94
    days_year = 252
    ewma_ann_vola = [None] * n
    ann_log_returns = [None]
    weights = []

    for i in range(n):
        weight = (1 - lambda_) * lambda_ ** i
        weights.append(weight)
    weights.reverse()
    weights = np.array(weights)

    var_t2_prev = None

    for t in range(1, len(data.index)):
        ann_log_return_t = np.log(data.iat[t] / data.iat[t - 1]) * days_year
        ann_log_returns.append(ann_log_return_t)

        if t >= n:
            mean_t = np.mean(ann_log_returns[-n:])
            var_t = sum(weights * (np.array(ann_log_returns[-n:]) - mean_t) ** 2) / n
            if var_t2_prev is None:
                var_t2 = var_t
            else:
                var_t2 = lambda_ * var_t2_prev + (1 - lambda_) * ann_log_return_t ** 2
            var_t2_prev = var_t2
            ewma_ann_vola.append(np.sqrt(var_t2))

    return pd.Series(ewma_ann_vola, index=data.index)


def gen_YZ_Vola(data: pd.DataFrame, days=14):
    days_year = 252

    RS_fac = [None]  # Rogers-Satchell
    ON_fac = [None]  # ON = overnight volatility
    OC_fac = [None]  # OC = open to close volatility
    sigma_YZ = [None] * days

    k = 0.34 / (1.34 + (days + 1) / (days - 1))

    for t in range(1, len(data.index)):
        RS_fac.append(np.log(data["High"].iat[t] / data["Close"].iat[t]) *
                      np.log(data["High"].iat[t] / data["Open"].iat[t]) +
                      np.log(data["Low"].iat[t] / data["Close"].iat[t]) *
                      np.log(data["Low"].iat[t] / data["Open"].iat[t]))

        ON_fac.append(np.log(data["Open"].iat[t] / data["Close"].iat[t - 1]))

        OC_fac.append(np.log(data["Close"].iat[t] / data["Open"].iat[t]))

        if t >= days:
            var_RS = days_year / days * np.sum(RS_fac[-days:])
            ON_mean = np.mean(ON_fac[-days:])
            var_ON = 1 / (days - 1) * np.sum((np.array(ON_fac[-days:]) - ON_mean) ** 2)
            OC_mean = np.mean(OC_fac[-days:])
            var_OC = 1 / (days - 1) * np.sum((np.array(OC_fac[-days:]) - OC_mean) ** 2)
            sigma_YZ_t = np.sqrt(days_year) * np.sqrt(var_ON + k * var_OC + (1 - k) * var_RS)
            sigma_YZ.append(sigma_YZ_t)

    return pd.Series(sigma_YZ, index=data.index)


def gen_binary_response(data: pd.DataFrame, returns):
    binary = [None]

    for t in range(1, len(returns)):
        if np.isnan(returns[t]):
            binary.append(None)
        elif returns[t] > 0:
            binary.append(1)
        else:
            binary.append(0)

    return pd.Series(binary, index=data.index)


def gen_tertiary_response(data: pd.DataFrame, returns, vola, days):
    tertiary = [None]

    upper_bound = 1/np.log(1 + days)
    mid_bound = upper_bound / 3

    for t in range(1, len(returns)):
        if np.isnan(vola[t]) or np.isnan(returns[t]):
            tertiary.append(None)
        elif returns[t] > mid_bound * vola[t]:
            tertiary.append(1)
        elif -mid_bound * vola[t] < returns[t] <= mid_bound * vola[t]:
            tertiary.append(0)
        elif returns[t] <= -mid_bound * vola[t]:
            tertiary.append(-1)
        else:
            raise ValueError("Invalid range for return: {}".format(returns[t]))

    return pd.Series(tertiary, index=data.index)


def gen_multinomial_response(data: pd.DataFrame, returns, vola, days):
    multinomial = [None]

    upper_bound = 1/np.log(1 + days)
    mid_bound = upper_bound / 3

    for t in range(1, len(returns)):
        if np.isnan(vola[t]) or np.isnan(returns[t]):
            multinomial.append(None)
        elif returns[t] > upper_bound * vola[t]:
            multinomial.append(2)
        elif mid_bound * vola[t] < returns[t] <= upper_bound * vola[t]:
            multinomial.append(1)
        elif -mid_bound * vola[t] < returns[t] <= mid_bound * vola[t]:
            multinomial.append(0)
        elif -upper_bound * vola[t] < returns[t] <= -mid_bound * vola[t]:
            multinomial.append(-1)
        elif returns[t] <= -upper_bound * vola[t]:
            multinomial.append(-2)
        else:
            raise ValueError("Invalid range for return: {}".format(returns[t]))

    return pd.Series(multinomial, index=data.index)


def gen_buy_sell_signals(asset, window_size=300, ema_fast="EMA_50", ema_slow="EMA_200"):
    signals = pd.Series(data=[0] * len(asset.data), index=asset.data.index)
    sell_signals = pd.Series(data=[np.nan] * len(asset.data), index=asset.data.index)
    buy_signals = pd.Series(data=[np.nan] * len(asset.data), index=asset.data.index)

    BUY = 1
    SELL = -1

    for i in range(len(asset.data) - window_size, -1, -window_size):
        min_value = None
        max_value = None
        min_date = None
        max_date = None
        for j in range(i, i + window_size):
            if (min_value is None or asset.data["Close"].iloc[j] < min_value) and \
                    asset.data["Close"].iloc[j] < asset.data[ema_slow].iloc[j] and \
                    asset.data["Close"].iloc[j] < asset.data[ema_fast].iloc[j]:
                min_value = asset.data["Close"].iloc[j]
                min_date = asset.data.index[j]
            if (max_value is None or asset.data["Close"].iloc[j] > max_value) and \
                    asset.data["Close"].iloc[j] > asset.data[ema_slow].iloc[j] and \
                    asset.data["Close"].iloc[j] > asset.data[ema_fast].iloc[j]:
                max_value = asset.data["Close"].iloc[j]
                max_date = asset.data.index[j]
        if min_date is not None:
            signals[min_date] = BUY
            buy_signals[min_date] = min_value
        if max_value is not None:
            signals[max_date] = SELL
            sell_signals[max_date] = max_value

    last_signal = None
    last_signal_date = None
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            if last_signal is not None:
                if last_signal == signals.iloc[i]:
                    signals[last_signal_date] = 0
                    buy_signals[last_signal_date] = None
                    sell_signals[last_signal_date] = None

            last_signal = signals.iloc[i]
            last_signal_date = signals.index[i]

    return signals, buy_signals, sell_signals


def col_postfix(col):
    if col[:4] == "EMA_":
        return "_E" + col[4:]
    return ""


def gen_indicators(asset):
    # Generate EMA indicator
    Log.info("Generating EMA...")
    for days in [5, 10, 20, 25, 50, 100, 150, 200]:
        EMA = gen_EMA(asset.data["Close"], n=days)
        asset.append("EMA_" + str(days), EMA)

    # Generate Buy/Sell signals
    Log.info("Generating Buy/Sell signals...")
    for days in [50, 100, 200, 300]:
        signal, buy_signal, sell_signal = gen_buy_sell_signals(asset, window_size=days)
        asset.append("Signal_" + str(days), signal)
        asset.append("Buy_signal_" + str(days), buy_signal)
        asset.append("Sell_signal_" + str(days), sell_signal)

    # Generate RSI indicator
    Log.info("Generating RSI...")
    for days, source in itertools.product([14, 28, 50], ["Close", "EMA_5"]):
        RSI = gen_RSI(asset.data[source], n=days)
        asset.append("RSI_" + str(days) + col_postfix(source), RSI)

    # Generate Stochastics K%D indicator
    Log.info("Generating Stochastics K%D...")
    for ((K, D, D2), source) in itertools.product([(5, 5, 3)], ["Close", "EMA_5"]):
        STOCH_K, STOCH_D, STOCH_D2 = \
            gen_Stochastics(asset.data[source], K_n=K, D=D, D2=D2)
        asset.append("STOCH_K" + col_postfix(source), STOCH_K)
        asset.append("STOCH_D_fast" + col_postfix(source), STOCH_D)
        asset.append("STOCH_D_slow" + col_postfix(source), STOCH_D2)

    # Generate MACD indicator
    Log.info("Generating MACD...")
    for source in ["Close", "EMA_5"]:
        MACD, MACD_Signal = gen_MACD(asset.data[source])
        asset.append("MACD" + col_postfix(source), MACD)
        asset.append("MACD_Signal" + col_postfix(source), MACD_Signal)

    # Generate CCI indicator
    Log.info("Generating CCI...")
    for days in [10, 20, 50]:
        CCI = gen_CCI(asset.data, n=days)
        asset.append("CCI_" + str(days), CCI)

    # Generate ATR indicator
    Log.info("Generating ATR...")
    for days in [10, 20, 50]:
        ATR = gen_ATR(asset.data, n=days)
        asset.append("ATR_" + str(days), ATR)

    # Generate Williams R indicator
    Log.info("Generating Williams R...")
    for (days, source) in itertools.product([10, 14, 20, 50], ["Close", "EMA_5"]):
        williams_r = gen_Williams_R(asset.data[source], n=days)
        asset.append("Williams_R_" + str(days) + col_postfix(source), williams_r)

    # Generate PROC
    Log.info("Generating Price Rate of Change...")
    for days, source in itertools.product([10, 14, 20, 40, 50, 60], ["Close", "EMA_5"]):
        proc = gen_price_rate_of_change(asset.data[source], n=days)
        asset.append("PROC_" + str(days) + col_postfix(source), proc)

    # Generate On Balance Volume
    Log.info("Generating On Balance Volume...")
    obv = gen_on_balance_volume(asset.data)
    asset.append("OBV", obv)

    # Generate ADL indicator
    Log.info("Generating ADL...")
    ADL = gen_ADL(asset.data)
    asset.append("ADL", ADL)

    # Generate returns
    Log.info("Generating returns...")
    for days in [1, 5, 20, 30, 40, 60, 90]:
        returns_d, log_returns_d, ann_log_returns_d, = gen_returns2(asset.data, delta=days)
        asset.append("returns_" + str(days), returns_d)
        asset.append("log_returns_" + str(days), log_returns_d)
        asset.append("ann_log_returns_" + str(days), ann_log_returns_d)

    # Generate simple volatility
    Log.info("Generating simple volatility...")
    for days in [1, 5, 20, 30, 40, 60, 90]:
        vola, ann_vola = gen_SimpleVola(asset.data["Close"], days=days)
        asset.append("vola_" + str(days), vola)
        asset.append("ann_vola_" + str(days), ann_vola)

    # Generate EWMA volatility
    Log.info("Generating EWMA volatility...")
    for days in [1, 5, 20, 30, 40, 60, 90]:
        EWMA_ann_vola = gen_EWMA_Vola(asset.data["Close"], n=max(days, 5))
        asset.append("EWMA_ann_vola_" + str(days), EWMA_ann_vola)
        tertiary_EWMA = gen_tertiary_response(asset.data,
                                              asset.data["ann_log_returns_" + str(days)],
                                              EWMA_ann_vola, days)
        asset.append("tertiary_EWMA_" + str(days), tertiary_EWMA)
        multinomial_EWMA = gen_multinomial_response(asset.data,
                                                    asset.data["ann_log_returns_" + str(days)],
                                                    EWMA_ann_vola, days)
        asset.append("multinomial_EWMA_" + str(days), multinomial_EWMA)

    # Generate Yang & Zhang volatility
    Log.info("Generating Yang & Zhang volatility...")
    for days in [1, 5, 20, 30, 40, 60, 90]:
        YZ_vola = gen_YZ_Vola(asset.data, days=max(days, 5))
        asset.append("YZ_Vola_" + str(days), YZ_vola)
        tertiary_YZ = gen_tertiary_response(asset.data,
                                            asset.data["ann_log_returns_" + str(days)],
                                            YZ_vola, days)
        asset.append("tertiary_YZ_" + str(days), tertiary_YZ)
        multinomial_YZ = gen_multinomial_response(asset.data,
                                                  asset.data["ann_log_returns_" + str(days)],
                                                  YZ_vola, days)
        asset.append("multinomial_YZ_" + str(days), multinomial_YZ)

    # Generate binary response variables
    for days in [1, 5, 20, 30, 40, 60, 90]:
        binary = gen_binary_response(asset.data, asset.data["ann_log_returns_" + str(days)])
        asset.append("binary_" + str(days), binary)

    return asset
