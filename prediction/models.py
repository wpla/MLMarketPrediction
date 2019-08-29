import itertools
import os
import re

import pandas as pd
import pycm
import sklearn
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, \
    AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

import indicators
from config import config
from log import Log


def make_data(asset, response_col, input_cols, window_size=10, days=None, flatten=False):
    y = []  # Response variable
    X = []  # Matrix of input variables

    response_var = asset.data[response_col]
    input_vars = asset.data[input_cols]
    input_len = len(asset.data[input_cols])

    if days is None:
        start = 0
    else:
        start = max([input_len - days, 0])

    for t in range(start, input_len, window_size):
        y_t = response_var[t - 1]
        X_t = input_vars[t - window_size:t]

        if not X_t.isna().any().any() and not pd.isna(y_t) and len(X_t) == window_size:
            y.append(y_t)
            if not flatten:
                X.append(X_t)
            else:
                X.append(X_t.values.flatten())

    return X, y


def split_data(X, y, test_factor=0.2):
    # Use the last 20% of the time series as test data.
    # The first elements contains the older data.
    test_len = int(len(X) * test_factor)
    X_test = X[-test_len:]
    y_test = y[-test_len:]
    X_train = X[:-test_len]
    y_train = y[:-test_len]
    return X_train, X_test, y_train, y_test


def make_response_col(response_var, response_param=None):
    if response_param is None:
        return response_var
    elif response_var in ['binary', 'tertiary_YZ', 'tertiary_EWMA', 'multinomial_YZ', 'multinomial_EWMA']:
        return response_var + "_" + str(response_param)
    return response_var


def create_response_data(asset, response_var, response_param=None):
    response_col = make_response_col(response_var, response_param)
    if response_col not in asset.data:
        Log.info("Need to make response variable: %s", response_col)
        if response_param is not None and "ann_log_returns_" + str(response_param) not in asset.data:
            returns_d, log_returns_d, ann_log_returns_d, = indicators.gen_returns2(asset.data, delta=response_param)
            asset.append("returns_" + str(response_param), returns_d)
            asset.append("log_returns_" + str(response_param), log_returns_d)
            asset.append("ann_log_returns_" + str(response_param), ann_log_returns_d)
        if response_var == "tertiary_YZ" and response_param is not None:
            YZ_vola = indicators.gen_YZ_Vola(asset.data, days=response_param)
            tertiary_YZ = indicators.gen_tertiary_response(asset.data, asset.data["ann_log_returns"], YZ_vola)
            asset.append(response_col, tertiary_YZ)
            Log.info("%s created", response_col)
        elif response_var == "multinomial_EWMA" and response_param is not None:
            EWMA_vola = indicators.gen_EWMA_Vola(asset.data["Close"], n=response_param)
            multinomial_EWMA = indicators.gen_multinomial_response(asset.data, asset.data["ann_log_returns"], EWMA_vola)
            asset.append(response_col, multinomial_EWMA)
            Log.info("%s created", response_col)
        elif response_var == "multinomial_EWMA" and response_param is not None:
            EWMA_vola = indicators.gen_EWMA_Vola(asset.data["Close"], n=response_param)
            multinomial_EWMA = indicators.gen_multinomial_response(asset.data, asset.data["ann_log_returns"], EWMA_vola)
            asset.append(response_col, multinomial_EWMA)
            Log.info("%s created", response_col)
        else:
            ValueError("Invalid response variable / response param: %s / %s", response_var, str(response_param))
    return asset


def make_input_col(input_var, days):
    if days is None:
        return input_var
    days = max(days, 10)
    for name in ["Close", "OBV", "ADL", "MACD", "STOCH"]:
        if name in input_var:
            return input_var
    for e in re.findall(r"_E\d+$", input_var):
        return input_var.replace(e, "_" + str(days) + e)
    return input_var + "_" + str(days)


def make_input_cols(input_vars, days):
    return [make_input_col(var, days) for var in input_vars]


def create_input_data(asset, input_vars, days=None):
    for input_var in input_vars:
        days = max(days, 10)
        input_col = make_input_col(input_var, days)

        source = "Close"  # We are calculating indicators from closing prices per default
        for ema_days in re.findall(r"_E(\d+)$", input_var):
            # if "_Exx" is given, we are calculating indicators from EMA prices
            source = "EMA_" + ema_days
            input_var = input_var.replace("_E" + ema_days, "")
            if source not in asset.data:
                EMA = indicators.gen_EMA(asset.data["Close"], n=int(ema_days))
                asset.append(source, EMA)
                Log.info("%s created", source)

        if input_col not in asset.data:
            Log.info("Need to make input variable %s (%s with parameter %d from %s)", input_col, input_var, days, source)
            if input_var == "RSI" and days is not None:
                RSI = indicators.gen_RSI(asset.data[source], days)
                asset.append(input_col, RSI)
                Log.info("%s created", input_col)
            elif input_var == "EMA" and days is not None:
                EMA = indicators.gen_EMA(asset.data["Close"], n=days)
                asset.append(input_col, EMA)
                Log.info("%s created", input_col)
            elif input_var == "CCI" and days is not None:
                CCI = indicators.gen_CCI(asset.data, days)
                asset.append(input_col, CCI)
                Log.info("%s created", input_col)
            elif input_var == "ATR" and days is not None:
                ATR = indicators.gen_ATR(asset.data, days)
                asset.append(input_col, ATR)
                Log.info("%s created", input_col)
            elif input_var == "ADL":
                ADL = indicators.gen_ADL(asset.data)
                asset.append(input_col, ADL)
                Log.info("%s created", input_col)
            elif input_var == "Williams_R":
                Williams_R = indicators.gen_Williams_R(asset.data[source], days)
                asset.append(input_col, Williams_R)
                Log.info("%s created", input_col)
            elif input_var == "PROC":
                PROC = indicators.gen_price_rate_of_change(asset.data[source], days)
                asset.append(input_col, PROC)
                Log.info("%s created", input_col)
            elif input_var == "OBV":
                OBV = indicators.gen_on_balance_volume(asset.data)
                asset.append(input_col, OBV)
                Log.info("%s created", input_col)
            else:
                ValueError("Invalid input variable / input param: %s / %s", input_var, str(days))

    return asset


def score_model(clf, X, y, split_factor=0.2, window_size=5):
    n = len(X)
    test_len = int(n * split_factor)

    y_real = []
    y_pred = []
    # y_proba = []

    for i in range(test_len, 0, -window_size):
        X_train = X[:n-i]
        y_train = y[:n-i]
        clf.fit(X_train, y_train)

        y_pred_t = clf.predict(X[n-i:n-i+window_size])
        # y_proba_t = clf.predict_proba(X[n-i:n-i+window_size])

        for y_t in y[n - i:n - i + window_size]:
            y_real.append(y_t)
        for y_t in y_pred_t:
            y_pred.append(y_t)
        # for y_t in y_proba_t:
        #     y_proba.append(y_t)

    return pycm.ConfusionMatrix(y_real, y_pred)


def gen_model_variants(response_vars, input_vars_list, days, window_sizes, model_params):
    for (response_var, input_vars, window_size, model_param) in itertools.product(
            response_vars, input_vars_list, window_sizes, model_params):

        if response_var in days:
            days_ = days[response_var]
        else:
            days_ = [None]

        for days__ in days_:
            for input_vars_n in range(len(input_vars)):
                for input_vars_ in itertools.combinations(input_vars, input_vars_n + 1):
                    yield response_var, input_vars_, days__, window_size, model_param


def fit_model(asset,
              model_name,
              create_func=None,
              use_scaler=False,
              response_vars=None,
              input_vars_list=None,
              days=None,
              window_sizes=None,
              model_params=None):
    Log.info("== Fitting model %s for %s", model_name, asset.symbol)

    if asset.symbol is None:
        asset.symbol = "(UNK)"

    if not os.path.exists(config().output_path()):
        os.mkdir(config().output_path())

    outfile = open(os.path.join(config().output_path(), model_name + "_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;"
                  "model_name;"
                  "response_col;"
                  "days;"
                  "input_cols;"
                  "window_size;"
                  "model_param;"
                  "n_train;"
                  "n_test;"
                  "Overall_MCC;"
                  "Overall_ACC;"
                  "PPV_Micro;"
                  "PPV_Macro;"
                  "TPR_Micro;"
                  "TPR_Macro;"
                  "Cramer's V;"
                  "P value\n")

    for response_var, input_vars, days_, window_size, model_param in gen_model_variants(response_vars, input_vars_list,
                                                                                        days, window_sizes,
                                                                                        model_params):
        # Create data
        asset = create_response_data(asset, response_var, days_)
        asset = create_input_data(asset, input_vars, days_)
        response_col = make_response_col(response_var, days_)
        input_cols = make_input_cols(input_vars, days_)

        Log.info("Fitting %s for response_col=%s, days=%s, input_cols=%s, "
                 "window_size=%s, model_param=%s", model_name, response_col, str(days_), str(input_cols),
                 str(window_size), str(model_param))

        # Create test and training data
        X, y = make_data(asset, response_col=response_col, input_cols=input_cols, window_size=window_size,
                         days=config().days(), flatten=True)

        X_train, X_test, y_train, y_test = split_data(X, y)

        Log.info("n_train: X: %d, y: %d", len(X_train), len(y_train))
        Log.info("n_test: X: %d, y: %d", len(X_test), len(y_test))

        if use_scaler:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        clf = create_func(model_param)
        scores = score_model(clf, X_train, y_train)
        Log.info("Model overall MCC: %0.4f" % scores.Overall_MCC)
        Log.info("Model overall ACC: %0.4f" % scores.Overall_ACC)

        # Write results to file
        outfile.write("%s;%s;%s;%s;%s;%s;%s;"
                      "%d;%d;"
                      "%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f\n" % (asset.symbol,
                                       model_name,
                                       response_col,
                                       days_,
                                       str(input_cols),
                                       str(window_size),
                                       str(model_param),
                                       len(X_train), len(X_test),
                                       scores.Overall_MCC,
                                       scores.Overall_ACC,
                                       scores.PPV_Micro,
                                       scores.PPV_Macro,
                                       scores.TPR_Micro,
                                       scores.TPR_Macro,
                                       scores.V,
                                       scores.PValue))
        outfile.flush()
    outfile.close()


def print_result_table(results):
    with open("results.txt", "w") as file:
        for method_name in results.keys():
            file.write("\\noindent " + method_name + " \n\n")

            file.write("\\begin{table}[]\n")
            file.write("\\begin{center}\n")
            file.write("\\begin{tabular}{@{}llrr@{}}\n")
            file.write("\\toprule\n")
            file.write("Exogenous variable & response & window size & score \\\\ \\midrule \n")

            first = True
            midrule = False
            for X_col in results[method_name].keys():

                if not first:
                    file.write("\\midrule\n")
                    midrule = False
                else:
                    first = False

                for y_col in results[method_name][X_col].keys():
                    if midrule:
                        file.write("\\cmidrule(l){2-4} \n")
                    for window_size in results[method_name][X_col][y_col].keys():
                        file.write("%s & %s & %d & %.2f \\\\ \n" %
                                   (X_col.replace("_", "\\_"), y_col.replace("_", "\\_"), window_size,
                                    results[method_name][X_col][y_col][window_size]["score"]))
                    midrule = True
            file.write("\\bottomrule\n")
            file.write("\\end{tabular}\n")
            file.write("\\end{center}\n")
            file.write("\\end{table}\n\n")


def create_linear_regression(model_param):
    return LinearRegression(**model_param)


def create_binary_logistic_regression(model_param):
    return LogisticRegression(**model_param)


def create_multinomial_logistic_regression(model_param):
    return LogisticRegression(multi_class='multinomial', **model_param)


def create_gaussian_naive_bayes(model_param):
    return GaussianNB()


def create_multinomial_naive_bayes(model_param):
    return MultinomialNB()


def create_decision_tree(model_param):
    return tree.DecisionTreeClassifier(**model_param)


def create_decision_tree_regressor(model_param):
    return tree.DecisionTreeRegressor(**model_param)


def create_bagging_dt(model_param):
    return BaggingClassifier(tree.DecisionTreeClassifier(), **model_param)


def create_bagging_dt_regressor(model_param):
    return BaggingRegressor(tree.DecisionTreeRegressor(), **model_param)


def create_random_forest(model_param):
    return RandomForestClassifier(**model_param)


def create_random_forest_regressor(model_param):
    return RandomForestRegressor(**model_param)


def create_adaboost(model_param):
    return AdaBoostClassifier(**model_param)


def create_adaboost_regressor(model_param):
    return AdaBoostRegressor(**model_param)


def create_SVM(model_param):
    return svm.SVC(**model_param)


def create_SVM_regressor(model_param):
    return svm.SVR(**model_param)


def create_KNN(model_param):
    return KNeighborsClassifier(**model_param)


def create_KNN_regressor(model_param):
    return KNeighborsRegressor(**model_param)


def create_MLP(model_param):
    return MLPClassifier(**model_param)


def create_MLP_regressor(model_param):
    return MLPRegressor(**model_param)


def fit_classifiers(asset, classifiers):
    days = {"binary": [1, 5, 20, 30, 60, 90],
            "tertiary_YZ": [1, 5, 20, 30, 60, 90],
            "multinomial_YZ": [1, 5, 20, 30, 60, 90],
            "multinomial_EWMA": [1, 5, 20, 30, 60, 90]}
    input_vars_list = [["RSI_E5", "STOCH_K", "MACD", "Williams_R_E5", "PROC_E5", "OBV", "CCI", "ATR", "ADL"],
                       ["RSI", "STOCH_K", "MACD", "Williams_R", "PROC", "OBV", "CCI", "ATR", "ADL"]
                      ]
    window_sizes = [1, 5, 10, 15, 20]

    # Logistic Regression
    if "logreg" in classifiers:
        models = {'LogReg-bin': ['binary'],
                  'LogReg-tert': ['tertiary_YZ'],
                  'LogReg-multi': ['multinomial_YZ']}
        model_params = [{'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 10000}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_binary_logistic_regression, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # Gaussian Naive Bayes
    if "nb" in classifiers:
        models = {'NaiveBayes-bin': ['binary'],
                  'NaiveBayes-tert': ['tertiary_YZ'],
                  'NaiveBayes-multi': ['multinomial_YZ']}
        model_params = [{}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_gaussian_naive_bayes, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # Decision Trees
    if "dt" in classifiers:
        models = {'DT-bin': ['binary'],
                  'DT-tert': ['tertiary_YZ'],
                  'DT-multi': ['multinomial_YZ']}
        model_params = [{'criterion': 'gini'},
                        {'criterion': 'entropy'}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_decision_tree, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # Bagging Decision Trees
    if "bdt" in classifiers:
        models = {'BaggingDT-bin': ['binary'],
                  'BaggingDT-tert': ['tertiary_YZ'],
                  'BaggingDT-multi': ['multinomial_YZ']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 50},
                        {'n_estimators': 100}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_bagging_dt, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # Random Forst Classifier
    if "rf" in classifiers:
        models = {'RF-bin': ['binary'],
                  'RF-tert': ['tertiary_YZ'],
                  'RF-multi': ['multinomial_YZ']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 50},
                        {'n_estimators': 100}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_random_forest, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # AdaBoost
    if "adaboost" in classifiers:
        models = {'AdaBoost-bin': ['binary'],
                  'AdaBoost-tert': ['tertiary_YZ'],
                  'AdaBoost-multi': ['multinomial_YZ']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 50},
                        {'n_estimators': 100}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_adaboost, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # SVM
    if "svm" in classifiers:
        models = {'SVM-bin': ['binary'],
                  'SVM-tert': ['tertiary_YZ'],
                  'SVM-multi': ['multinomial_YZ']}
        model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                        {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_SVM, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # KNN
    if "knn" in classifiers:
        models = {'KNN-bin': ['binary'],
                  'KNN-tert': ['tertiary_YZ'],
                  'KNN-multi': ['multinomial_YZ']}
        model_params = [{'n_neighbors': 5},
                        {'n_neighbors': 10}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_KNN, use_scaler=False,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)

    # ANN
    if "nn" in classifiers:
        models = {'ANN-bin': ['binary'],
                  'ANN-tert': ['tertiary_YZ'],
                  'ANN-multi': ['multinomial_YZ']}
        model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 50, 10)}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_MLP, use_scaler=True,
                      response_vars=response_vars, input_vars_list=input_vars_list, days=days,
                      window_sizes=window_sizes, model_params=model_params)


def fit_regressors(asset, regressors):
    input_vars_list = [('Close',),
                       ('EMA',),
                       ('returns',),
                       ('log_returns',),
                       ("RSI",),
                       ("Stochastic",),
                       ("MACD",),
                       ("CCI",),
                       ("ATR",),
                       ("ADL",),
                       ("EMA", "RSI"),
                       ("EMA", "RSI", "STOCH_K"),
                       ("EMA", "RSI", "MACD"),
                       ("EMA", "RSI", "CCI"),
                       ("EMA", "RSI", "ATR"),
                       ]
    window_sizes = [1, 5, 10, 15, 21]

    # Linear Regression
    if "linreg" in regressors:
        response_vars = ['log_returns']
        model_params = [{}]
        fit_model(asset, "LinRegr", create_func=create_linear_regression, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Decision Tree Regressor
    if "dt" in regressors:
        response_vars = ['log_returns']
        model_params = [{'max_depth': 10},
                        {'max_depth': 30}]
        fit_model(asset, "DTRegr", create_func=create_decision_tree_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Bagging Decision Trees Regressors
    if "bdt" in regressors:
        response_vars = ['log_returns']
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        fit_model(asset, "BaggingDTRegr", create_func=create_bagging_dt_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Random Forst Regressor
    if "rf" in regressors:
        response_vars = ['log_returns']
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        fit_model(asset, "RFRegr", create_func=create_random_forest_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # AdaBoostRegressor
    if "adaboost" in regressors:
        response_vars = ['log_returns']
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        fit_model(asset, "AdaBoostRegr", create_func=create_adaboost_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # SVM Regressor
    if "svm" in regressors:
        response_vars = ['log_returns']
        model_params = [{'kernel': 'rbf', 'gamma': 'auto'}]
        fit_model(asset, "SVMRegr", create_func=create_SVM_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # KNN Regressor
    if "knn" in regressors:
        response_vars = ['log_returns']
        model_params = [{'n_neighbors': 5},
                        {'n_neighbors': 10}]
        fit_model(asset, "KNNRegr", create_func=create_KNN_regressor, use_scaler=False,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)

    # ANN Regressor
    if "nn" in regressors:
        response_vars = ['log_returns']
        model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 20, 3)}]
        fit_model(asset, "ANNRegr", create_func=create_MLP_regressor, use_scaler=True,
                  model_params=model_params, response_vars=response_vars,
                  input_vars_list=input_vars_list, window_sizes=window_sizes)
