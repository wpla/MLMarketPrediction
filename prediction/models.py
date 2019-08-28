import itertools
import os

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

    return y, X


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
        if response_var == "multinomial_YZ" and response_param is not None:
            YZ_vola = indicators.gen_YZ_Vola(asset.data, days=response_param)
            multinomial_YZ = indicators.gen_multinomial_response(asset.data, asset.data["ann_log_returns"], YZ_vola)
            asset.append(response_col, multinomial_YZ)
            Log.info("%s created", response_col)
        elif response_var == "multinomial_EWMA" and response_param is not None:
            EWMA_vola = indicators.gen_EWMA_Vola(asset.data["Close"], n=response_param)
            multinomial_EWMA = indicators.gen_multinomial_response(asset.data, asset.data["ann_log_returns"], EWMA_vola)
            asset.append(response_col, multinomial_EWMA)
            Log.info("%s created", response_col)
        else:
            ValueError("Invalid response variable / response param: %s / %s", response_var, str(response_param))
    return asset


def make_input_col(input_var, input_param):
    if input_param is None:
        return input_var
    elif input_var in ["Close", "returns", "log_returns", "ann_log_returns"]:
        return input_var
    return input_var + "_" + str(input_param)


def create_input_data(asset, input_var, input_param=None):
    input_col = make_input_col(input_var, input_param)

    if input_col not in asset.data:
        Log.info("Need to make input variable: %s", input_col)
        if input_var == "RSI" and input_param is not None:
            RSI = indicators.gen_RSI(asset.data["Close"], input_param)
            asset.append(input_col, RSI)
            Log.info("%s created", input_col)
        elif input_var == "EMA" and input_param is not None:
            EMA = indicators.gen_EMA(asset.data["Close"], n=input_param)
            asset.append(input_col, EMA)
            Log.info("%s created", input_col)
        elif input_var == "STOCH" and input_param is not None:
            K, _, D = indicators.gen_Stochastics(asset.data["Close"], K_n=input_param)
            asset.append(input_col, K)
            asset.append(input_col + "_D", D)
            Log.info("%s created", input_col)
        elif input_var == "MACD" and input_param is not None:
            MACD, Signal = indicators.gen_MACD(asset.data["Close"], input_param)
            asset.append(input_col, MACD)
            asset.append(input_col + "_Signal", Signal)
            Log.info("%s created", input_col)
        elif input_var == "CCI" and input_param is not None:
            CCI = indicators.gen_CCI(asset.data, input_param)
            asset.append(input_col, CCI)
            Log.info("%s created", input_col)
        elif input_var == "ATR" and input_param is not None:
            ATR = indicators.gen_ATR(asset.data, input_param)
            asset.append(input_col, ATR)
            Log.info("%s created", input_col)
        elif input_var == "ADL":
            ADL = indicators.gen_ADL(asset.data)
            asset.append(input_col, ADL)
            Log.info("%s created", input_col)
        else:
            ValueError("Invalid input variable / input param: %s / %s", input_var, str(input_param))

    return asset


def score_model(clf, X, y, split_factor=0.2):
    n = len(X)
    test_len = int(n * split_factor)

    y_real = []
    y_pred = []
    y_proba = []

    for i in range(test_len, 0, -1):
        X_train = X[:-i]
        y_train = y[:-i]
        clf.fit(X_train, y_train)

        y_pred_t = clf.predict([X[n - i]])[0]
        y_proba_t = clf.predict_proba([X[n - i]])[0]

        y_real.append(y[n - i])
        y_pred.append(y_pred_t)
        y_proba.append(y_proba_t)

    return pycm.ConfusionMatrix(y_real, y_pred)


def fit_model(asset,
              model_name,
              create_func=None,
              model_params=None,
              use_scaler=False,
              response_vars=None,
              response_params=None,
              input_vars_list=None,
              window_sizes=None):
    Log.info("== Fitting model %s for %s", model_name, asset.symbol)

    model_scores = {}

    if asset.symbol is None:
        asset.symbol = "_UNK_"

    outfile = open(os.path.join(config().output_path(), model_name + "_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;model;response_var;response_param;input_var;input_param;window_size;model_param;"
                  "n_train;n_test;Overall_MCC;test_score\n")

    for (response_var, input_vars, window_size, model_param) in itertools.product(
            response_vars, input_vars_list, window_sizes, model_params):

        if response_var in response_params:
            response_params_ = response_params[response_var]
        else:
            response_params_ = [None]

        for response_param in response_params_:
            Log.info("Fitting %s for response_var=%s, respone_param=%s, input_var=%s, "
                     "window_size=%s, model_param=%s", model_name, response_var, str(response_param), str(input_vars),
                     str(window_size), str(model_param))

            # Create data
            # asset = create_response_data(asset, response_var, response_param)
            # asset = create_input_data(asset, input_var, input_param)
            response_col = make_response_col(response_var, response_param)
            # input_col = make_input_col(input_var, input_param)

            # Create test and training data
            y, X = make_data(asset, response_col=response_col, input_cols=input_vars, window_size=window_size,
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
            Log.info("Model overall MCC: %0.2f" % scores.Overall_MCC)

            # Write results to file
            outfile.write("%s;%s;%s;%s;%s;%s;%s;"
                          "%d;%d;"
                          "%.4f;%.4f\n" % (asset.symbol, model_name,
                                           response_var, response_param,
                                           str(input_vars),
                                           str(window_size),
                                           str(model_param),
                                           len(X_train), len(X_test),
                                           scores.Overall_MCC,
                                           0.0))
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
    response_params = {"binary": [1, 5, 20, 30, 60, 90],
                       "tertiary_YZ": [1, 5, 20, 30, 60, 90],
                       "multinomial_YZ": [1, 5, 20, 30, 60, 90],
                       "multinomial_EWMA": [1, 5, 20, 30, 60, 90]}
    input_vars_list = [['Close'],
                       ['EMA'],
                       ['returns'],
                       ['log_returns'],
                       ["RSI"],
                       ["STOCH_K"],
                       ["MACD"],
                       ["CCI"],
                       ["ATR"],
                       ["ADL"],
                       ["EMA", "RSI"],
                       ["EMA", "RSI", "STOCH_K"],
                       ["EMA", "RSI", "MACD"],
                       ["EMA", "RSI", "CCI"],
                       ["EMA", "RSI", "ATR"],
                       ]
    window_sizes = [1, 5, 10, 15, 21]

    # Logistic Regression
    if "logreg" in classifiers:
        models = {'LogReg-bin': ['binary'],
                  'LogReg-tert': ['tertiary_YZ'],
                  'LogReg-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 10000}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_binary_logistic_regression, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Gaussian Naive Bayes
    if "nb" in classifiers:
        models = {'NaiveBayes-bin': ['binary'],
                  'NaiveBayes-tert': ['tertiary_YZ'],
                  'NaiveBayes-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_gaussian_naive_bayes, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Decision Trees
    if "dt" in classifiers:
        models = {'DT-bin': ['binary'],
                  'DT-tert': ['tertiary_YZ'],
                  'DT-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'criterion': 'gini'},
                        {'criterion': 'entropy'}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_decision_tree, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Bagging Decision Trees
    if "bdt" in classifiers:
        models = {'BaggingDT-bin': ['binary'],
                  'BaggingDT-tert': ['tertiary_YZ'],
                  'BaggingDT-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_bagging_dt, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # Random Forst Classifier
    if "rf" in classifiers:
        models = {'RF-bin': ['binary'],
                  'RF-tert': ['tertiary_YZ'],
                  'RF-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_random_forest, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # AdaBoost
    if "adaboost" in classifiers:
        models = {'AdaBoost-bin': ['binary'],
                  'AdaBoost-tert': ['tertiary_YZ'],
                  'AdaBoost-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'n_estimators': 10},
                        {'n_estimators': 100},
                        {'n_estimators': 300}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_adaboost, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # SVM
    if "svm" in classifiers:
        models = {'SVM-bin': ['binary'],
                  'SVM-tert': ['tertiary_YZ'],
                  'SVM-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                        {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_SVM, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # KNN
    if "knn" in classifiers:
        models = {'KNN-bin': ['binary'],
                  'KNN-tert': ['tertiary_YZ'],
                  'KNN-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'n_neighbors': 5},
                        {'n_neighbors': 10}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_KNN, use_scaler=False,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)

    # ANN Binomial
    if "nn" in classifiers:
        models = {'ANN-bin': ['binary'],
                  'ANN-tert': ['tertiary_YZ'],
                  'ANN-multi': ['multinomial_YZ', 'multinomial_EWMA']}
        model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)},
                        {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 20, 3)}]
        for model_name, response_vars in models.items():
            fit_model(asset, model_name, create_func=create_MLP, use_scaler=True,
                      model_params=model_params, response_vars=response_vars, response_params=response_params,
                      input_vars_list=input_vars_list, window_sizes=window_sizes)


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
