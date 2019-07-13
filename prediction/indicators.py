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


def gen_Stochastics(data: pd.Series, K_n=14, D_n=3):
    K = [0] * K_n
    D_fast = [0] * K_n
    D_slow = [0] * K_n
    for t in range(K_n, len(data.index)):
        L = min(data[(t + 1 - K_n):(t + 1)])
        H = max(data[(t + 1 - K_n):(t + 1)])
        K_t = 100 * (data.iat[t] - L) / (H - L)
        K.append(K_t)
        D_fast_t = sum(K[-D_n:]) / D_n
        D_fast.append(D_fast_t)
        D_slow_t = sum(D_fast[-D_n:]) / D_n
        D_slow.append(D_slow_t)
    return pd.Series(K, index=data.index), \
           pd.Series(D_fast, index=data.index), \
           pd.Series(D_slow, index=data.index)


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


def gen_multinomial_response(data: pd.DataFrame, returns, vola):
    multinomial = [None]

    upper_bound = 1.4
    mid_bound = upper_bound / 3

    for t in range(1, len(returns)):
        if np.isnan(vola[t]):
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


def gen_indicators(asset):
    # Generate EMA indicator
    Log.info("Generating EMA...")
    EMA_5 = gen_EMA(asset.data["Close"], n=5)
    EMA_10 = gen_EMA(asset.data["Close"], n=10)
    EMA_20 = gen_EMA(asset.data["Close"], n=20)
    EMA_50 = gen_EMA(asset.data["Close"], n=50)
    asset.append("EMA_5", EMA_5)
    asset.append("EMA_10", EMA_10)
    asset.append("EMA_20", EMA_20)
    asset.append("EMA_50", EMA_50)

    # Generate RSI indicator
    Log.info("Generating RSI...")
    RSI_10 = gen_RSI(asset.data["Close"], n=10)
    RSI_20 = gen_RSI(asset.data["Close"], n=20)
    RSI_50 = gen_RSI(asset.data["Close"], n=50)
    asset.append("RSI_10", RSI_10)
    asset.append("RSI_20", RSI_20)
    asset.append("RSI_50", RSI_50)

    # Generate Stochastics K%D indicator
    Log.info("Generating Stochastics K%D...")
    Stochastics_K_14, Stochastics_D_fast, Stochastics_D_slow = \
        gen_Stochastics(asset.data["Close"], K_n=14)
    asset.append("Stochastics_K_14", Stochastics_K_14)
    asset.append("Stochastics_D_fast", Stochastics_D_fast)
    asset.append("Stochastics_D_slow", Stochastics_D_slow)

    # Generate MACD indicator
    Log.info("Generating MACD...")
    MACD, MACD_Signal = gen_MACD(asset.data["Close"])
    asset.append("MACD(12,26,9)", MACD)
    asset.append("MACD_Signal(12,26,9)", MACD_Signal)

    # Generate CCI indicator
    Log.info("Generating CCI...")
    CCI_10 = gen_CCI(asset.data, n=10)
    CCI_20 = gen_CCI(asset.data, n=20)
    CCI_50 = gen_CCI(asset.data, n=50)
    asset.append("CCI_10", CCI_10)
    asset.append("CCI_20", CCI_20)
    asset.append("CCI_50", CCI_50)

    # Generate ATR indicator
    Log.info("Generating ATR...")
    ATR_10 = gen_ATR(asset.data, n=10)
    ATR_20 = gen_ATR(asset.data, n=20)
    ATR_50 = gen_ATR(asset.data, n=50)
    asset.append("ATR_10", ATR_10)
    asset.append("ATR_20", ATR_20)
    asset.append("ATR_50", ATR_50)

    # Generate ADL indicator
    Log.info("Generating ADL...")
    ADL = gen_ADL(asset.data)
    asset.append("ADL", ADL)

    # Generate returns
    Log.info("Generating returns...")
    returns, log_returns, ann_log_returns, mon_log_returns, qu_log_return, yearly_log_returns = \
        gen_returns(asset.data)

    asset.append("returns", returns)
    asset.append("log_returns", log_returns)
    asset.append("ann_log_returns", ann_log_returns)
    asset.append("monthly_log_returns", mon_log_returns)
    asset.append("quarterly_log_returns", qu_log_return)
    asset.append("yearly_log_returns", yearly_log_returns)

    # Generate simple volatility
    Log.info("Generating simple volatility...")
    vola_10, ann_vola_10 = gen_SimpleVola(asset.data["Close"], days=10)
    vola_20, ann_vola_20 = gen_SimpleVola(asset.data["Close"], days=20)
    asset.append("vola_10", vola_10)
    asset.append("ann_vola_10", ann_vola_10)
    asset.append("vola_20", vola_20)
    asset.append("ann_vola_20", ann_vola_20)

    # Generate EWMA volatility
    Log.info("Generating EWMA volatility...")
    EWMA_ann_vola_10 = gen_EWMA_Vola(asset.data["Close"], n=10)
    EWMA_ann_vola_20 = gen_EWMA_Vola(asset.data["Close"], n=20)
    asset.append("EWMA_ann_vola_10", EWMA_ann_vola_10)
    asset.append("EWMA_ann_vola_20", EWMA_ann_vola_20)

    # Generate Yang & Zhang volatility
    Log.info("Generating Yang & Zhang volatility...")
    YZ_vola_10 = gen_YZ_Vola(asset.data, days=10)
    YZ_vola_20 = gen_YZ_Vola(asset.data, days=20)
    asset.append("YZ_Vola_10", YZ_vola_10)
    asset.append("YZ_Vola_20", YZ_vola_20)

    # Generate binary and multinomial response variables
    Log.info("Generating response variables...")
    binary = gen_binary_response(asset.data, ann_log_returns)
    multinomial_YZ_10 = gen_multinomial_response(asset.data, ann_log_returns, YZ_vola_10)
    multinomial_EWMA_10 = gen_multinomial_response(asset.data, ann_log_returns, EWMA_ann_vola_10)
    multinomial_YZ_20 = gen_multinomial_response(asset.data, ann_log_returns, YZ_vola_20)
    multinomial_EWMA_20 = gen_multinomial_response(asset.data, ann_log_returns, EWMA_ann_vola_20)

    asset.append("binary", binary)
    asset.append("multinomial_YZ_10", multinomial_YZ_10)
    asset.append("multinomial_YZ_20", multinomial_YZ_20)
    asset.append("multinomial_EWMA_10", multinomial_EWMA_10)
    asset.append("multinomial_EWMA_20", multinomial_EWMA_20)

    return asset


