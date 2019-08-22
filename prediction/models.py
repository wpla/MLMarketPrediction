import itertools
import os

import pandas as pd
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


def make_data(asset, response_col, input_col, window_size=10, days=5000):
    y = []  # endogenous variable
    X = []  # Matrix of exogenous variables

    response_var = asset.data[response_col]
    input_vars = asset.data[input_col]
    input_len = len(asset.data[input_col])

    for t in range(input_len - 1, max([input_len - days, 0]), -(window_size + 1)):
        y_t = response_var[t]
        X_t = input_vars[t - window_size:t]

        if not X_t.isna().any() and not pd.isna(y_t) and len(X_t) == window_size:
            y.append(y_t)
            X.append(X_t)

    return y, X


def make_data_multicol(asset, response_col, input_cols, window_size=10, days=5000):
    y = []  # Response variable
    X = []  # Matrix of input variables

    response_var = asset.data[response_col]
    input_vars = asset.data[input_cols]
    input_len = len(asset.data[input_cols])

    for t in range(input_len - 1, max([input_len - days, 0]), -(window_size + 1)):
        y_t = response_var[t]
        X_t = input_vars[t - window_size:t]

        if not X_t.isna().any().any() and not pd.isna(y_t) and len(X_t) == window_size:
            y.append(y_t)
            X.append(X_t)

    return y, X


def split_data(X, y, test_factor=0.2):
    # Use the first 20% of the time series as test data.
    # The first elements contains the newer data.
    test_len = int(len(X) * test_factor)
    X_test = X[:test_len]
    y_test = y[:test_len]
    X_train = X[test_len:]
    y_train = y[test_len:]
    return X_train, X_test, y_train, y_test


def make_response_col(response_var, response_param=None):
    if response_param is None:
        return response_var
    elif response_var in ['multinomial_YZ', 'multinomial_EWMA']:
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
        elif input_var == "Stochastic" and input_param is not None:
            K, _, D = indicators.gen_Stochastics(asset.data["Close"], K=input_param)
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


def fit_cross_validation(asset,
                         model_name,
                         create_clf=None,
                         model_params=None,
                         use_scaler=False,
                         response_vars=None,
                         response_params=None,
                         input_vars=None,
                         input_params=None,
                         window_sizes=None,
                         use_test_train_split=False):
    Log.info("== Fitting model %s for %s", model_name, asset.symbol)

    model_scores = {}

    if asset.symbol is None:
        asset.symbol = "_UNK_"

    outfile = open(os.path.join(config().output_path(), model_name + "_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;model;response_var;response_param;input_var;input_param;window_size;model_param;"
                  "n_train;n_test;cv_score_mean;cv_score_std;test_score\n")

    for (response_var, input_var, window_size, model_param) in itertools.product(
            response_vars, input_vars, window_sizes, model_params):

        if response_var in response_params:
            response_params_ = response_params[response_var]
        else:
            response_params_ = [None]

        if input_var in input_params:
            input_params_ = input_params[input_var]
        else:
            input_params_ = [None]

        for response_param in response_params_:
            for input_param in input_params_:

                Log.info("Fitting %s for response_var=%s, respone_param=%s, input_var=%s, input_param=%s, "
                         "window_size=%s, model_param=%s", model_name, response_var, str(response_param), input_var,
                         str(input_param), str(window_size), str(model_param))

                # Create data
                asset = create_response_data(asset, response_var, response_param)
                asset = create_input_data(asset, input_var, input_param)
                response_col = make_response_col(response_var, response_param)
                input_col = make_input_col(input_var, input_param)

                # Create test and training data
                y, X = make_data(asset, response_col=response_col, input_col=input_col, window_size=window_size,
                                 days=config().days())

                # Split data into training and test data
                if use_test_train_split:
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = split_data(X, y)

                Log.info("n_train: X: %d, y: %d", len(X_train), len(y_train))
                Log.info("n_test: X: %d, y: %d", len(X_test), len(y_test))

                if use_scaler:
                    scaler = StandardScaler()
                    scaler.fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                clf = create_clf(model_param)
                scores = cross_val_score(clf, X_train, y_train, cv=5)
                Log.info("Model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

                clf2 = create_clf(model_param)
                clf2.fit(X_train, y_train)
                test_score = clf2.score(X_test, y_test)
                Log.info("Model test accuracy: %0.2f" % test_score)

                # Write results to file
                outfile.write("%s;%s Training;%s;%s;%s;%s;%s;%s;"
                              "%d;%d;"
                              "%.4f;%.4f;%.4f\n" % (asset.symbol, model_name,
                                                    response_var, response_param,
                                                    input_var,
                                                    input_param,
                                                    str(window_size),
                                                    str(model_param),
                                                    len(X_train), len(X_test),
                                                    scores.mean(), scores.std(),
                                                    test_score))

                # Store parameters and accuracy
                model_scores[(response_var, response_param, input_var, input_param, window_size, str(model_param))] = (
                    scores.mean(), scores.std(), test_score)

    # Search for model with best score
    best_cv_score = None
    best_cv_score_std = None
    test_score = None
    best_params = None
    for p in model_scores.keys():
        score_mean, score_std, test_score_ = model_scores[p]
        if best_cv_score is None or best_cv_score < score_mean:
            best_cv_score = score_mean
            best_cv_score_std = score_std
            test_score = test_score_
            best_params = p

    # Output winner model
    if best_cv_score is not None:
        Log.info("== Best model ==")
        response_var, response_param, input_var, input_param, window_size, model_param_str = best_params
        Log.info("Parameter: response_var=%s, respone_param=%s, input_var=%s, input_param=%s, window_size=%s, "
                 "model_param=%s", response_var, str(response_param), input_var, str(input_param), str(window_size),
                 model_param_str)
        Log.info("Cross validation accuracy: %0.2f (+/- %0.2f)" % (best_cv_score, best_cv_score_std))
        Log.info("Test accuracy: %0.2f " % test_score)

        # Write winner to file
        outfile.write("%s;Winner %s;%s;%s;%s;%s;%s;%s;%.4f;%.4f;%.4f\n" % (
            asset.symbol, model_name, response_var, response_param, input_var, input_param, str(window_size),
            model_param_str,
            best_cv_score, best_cv_score_std, test_score))
    else:
        Log.warn("No best model found")
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


def fit_models(asset):
    results = {}
    for col in ["ann_log_returns", "RSI_10", "Stochastics_K_10",
                "MACD(12,26,9)", "MACD_Signal(12,26,9)",
                "CCI_10", "ATR_10", "ADL"]:
        Log.info("Selecting column: %s", col)
        for window_size in [1, 5, 10, 15, 21]:
            results = fit_binary_logistic_regression(asset, "binary", col, window_size, results)
            results = fit_multinomial_logistic_regression(asset, "multinomial_EWMA", col, window_size, results)
            results = fit_multinomial_logistic_regression(asset, "multinomial_YZ", col, window_size, results)

            results = fit_naive_bayes(asset, "multinomial_YZ", col, window_size, results)
            results = fit_multinomial_naive_bayes(asset, "multinomial_YZ", col, window_size, results)
            results = fit_support_vector_machines(asset, "multinomial_YZ", col, window_size, results)
            results = fit_KNN(asset, "multinomial_YZ", col, window_size, results)
            results = fit_decision_trees(asset, "multinomial_YZ", col, window_size, results)
            results = fit_adaboost(asset, "multinomial_YZ", col, window_size, results)
            results = fit_bagging_logistic_regression(asset, "multinomial_YZ", col, window_size, results)
            results = fit_ANN(asset, "multinomial_YZ", col, window_size, results)

    print_result_table(results)


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


def fit_models_crossvalidated(asset):
    response_params = {'multinomial_YZ': [10, 20],
                       'multinomial_EWMA': [10, 20]
                       }
    input_vars = ["Close", "EMA", "returns", "log_returns", "ann_log_returns",
                  "RSI", "Stochastic", "MACD", "CCI", "ATR", "ADL"]
    input_params = {"EMA": [5, 10, 20, 50],
                    "RSI": [5, 10, 20, 50],
                    "Stochastic": [5, 10, 20, 50],
                    "MACD": [5, 10, 20, 50],
                    "CCI": [5, 10, 20, 50],
                    "ATR": [5, 10, 20, 50]
                    }
    window_sizes = [1, 5, 10, 15, 21]

    response_params = {"binary": [1, 5, 20, 30, 60, 90],
                       "multinomial_YZ": [1, 5, 20, 30, 60, 90],
                       "tertiary_YZ": [1, 5, 20, 30, 60, 90]}
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
                       ("EMA", "RSI", "Stochastic"),
                       ("EMA", "RSI", "MACD"),
                       ("EMA", "RSI", "CCI"),
                       ("EMA", "RSI", "ATR"),
                       ]
    window_sizes = [1, 5, 10, 15, 21]

    # Linear Regression
    response_vars = ['log_returns']
    model_params = [{}]
    fit_cross_validation(asset, "LinRegr", create_clf=create_linear_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Binary Logistic Regression
    response_vars = ['binary']
    model_params = [{'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 10000}]
    fit_cross_validation(asset, "BinLogRegr", create_clf=create_binary_logistic_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Multinomial Logistic Regression
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'solver': 'lbfgs', 'max_iter': 10000}]
    fit_cross_validation(asset, "MultiLogRegr", create_clf=create_multinomial_logistic_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes
    response_vars = ['binary']
    model_params = [{}]
    fit_cross_validation(asset, "GaussNBBin", create_clf=create_gaussian_naive_bayes, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{}]
    fit_cross_validation(asset, "GaussNBMulti", create_clf=create_gaussian_naive_bayes, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes Regressor
    # response_vars = ['log_returns']
    # model_params = [{}]
    # fit_cross_validation(asset, "GaussNBRegr", create_clf=create_gaussian_naive_bayes, use_scaler=False,
    #                      model_params=model_params, response_vars=response_vars, response_params=response_params,
    #                      input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Multinomial Naive Bayes
    # response_vars = ['multinomial_YZ']
    # model_params = [{}]
    # fit_cross_validation(asset, "Multi_NB", create_clf=create_multinomial_naive_bayes, use_scaler=False,
    #                      model_params=model_params, response_vars=response_vars, response_params=response_params,
    #                      input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Decision Trees
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'criterion': 'gini'},
                    {'criterion': 'entropy'}]
    fit_cross_validation(asset, "DT", create_clf=create_decision_tree, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Decision Tree Regressor
    response_vars = ['log_returns']
    model_params = [{'max_depth': 10},
                    {'max_depth': 30}]
    fit_cross_validation(asset, "DTRegr", create_clf=create_decision_tree_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Bagging Decision Trees
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "BaggingDT", create_clf=create_bagging_dt, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Bagging Decision Trees Regressors
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "BaggingDTRegr", create_clf=create_bagging_dt_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Random Forst Classifier
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "RF", create_clf=create_random_forest, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Random Forst Regressor
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "RFRegr", create_clf=create_random_forest_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # AdaBoost
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "AdaBoost", create_clf=create_adaboost, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # AdaBoostRegressor
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 10},
                    {'n_estimators': 100},
                    {'n_estimators': 300}]
    fit_cross_validation(asset, "AdaBoostRegr", create_clf=create_adaboost_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM
    response_vars = ['binary']
    model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                    {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMBin", create_clf=create_SVM, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                    {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMMulti", create_clf=create_SVM, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM Regressor
    response_vars = ['log_returns']
    model_params = [{'kernel': 'rbf', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMRegr", create_clf=create_SVM_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # KNN
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'n_neighbors': 5},
                    {'n_neighbors': 10}]
    fit_cross_validation(asset, "KNN", create_clf=create_KNN, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # KNN Regressor
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'n_neighbors': 5},
                    {'n_neighbors': 10}]
    fit_cross_validation(asset, "KNNRegr", create_clf=create_KNN_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Binomial
    response_vars = ['binary']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 20, 3)}]
    fit_cross_validation(asset, "ANNBinary", create_clf=create_MLP, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Multinomial
    response_vars = ['multinomial_YZ', 'multinomial_EWMA']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 20, 3)}]
    fit_cross_validation(asset, "ANNMulti", create_clf=create_MLP, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Regressor
    response_vars = ['log_returns']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (100, 20, 3)}]
    fit_cross_validation(asset, "ANNRegr", create_clf=create_MLP_regressor, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)


def fit_models_crossvalidated_test(asset):
    response_params = {'multinomial_YZ': [10]}
    input_vars = ["Close", "ann_log_returns", "RSI", "CCI"]
    input_params = {"RSI": [10, 20],
                    "Stochastic": [10, 20],
                    "MACD": [10, 20],
                    "CCI": [10, 20],
                    "ATR": [10, 20]
                    }
    window_sizes = [5, 10]

    Log.info("Using %d days of data.", config().days())

    # Regression
    response_vars = ['log_returns']
    model_params = [{}]
    fit_cross_validation(asset, "LinRegr", create_clf=create_linear_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Binary Logistic Regression
    response_vars = ['binary']
    model_params = [{'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}]
    fit_cross_validation(asset, "BinLogRegr", create_clf=create_binary_logistic_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Multinomial Logistic Regression
    response_vars = ['multinomial_YZ']
    model_params = [{'solver': 'lbfgs', 'max_iter': 1000}]
    fit_cross_validation(asset, "MultiLogRegr", create_clf=create_multinomial_logistic_regression, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes
    response_vars = ['binary']
    model_params = [{}]
    fit_cross_validation(asset, "GaussNBBin", create_clf=create_gaussian_naive_bayes, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes
    response_vars = ['multinomial_YZ']
    model_params = [{}]
    fit_cross_validation(asset, "GaussNBMulti", create_clf=create_gaussian_naive_bayes, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Gaussian Naive Bayes Regressor
    # response_vars = ['log_returns']
    # model_params = [{}]
    # fit_cross_validation(asset, "GaussNBRegr", create_clf=create_gaussian_naive_bayes, use_scaler=False,
    #                      model_params=model_params, response_vars=response_vars, response_params=response_params,
    #                      input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Multinomial Naive Bayes
    # response_vars = ['multinomial_YZ']
    # model_params = [{}]
    # fit_cross_validation(asset, "Multi_NB", create_clf=create_multinomial_naive_bayes, use_scaler=False,
    #                      model_params=model_params, response_vars=response_vars, response_params=response_params,
    #                      input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Decision Trees
    response_vars = ['multinomial_YZ']
    model_params = [{'criterion': 'gini'},
                    {'criterion': 'entropy'}]
    fit_cross_validation(asset, "DT", create_clf=create_decision_tree, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Decision Tree Regressor
    response_vars = ['log_returns']
    model_params = [{'max_depth': 10}]
    fit_cross_validation(asset, "DTRegr", create_clf=create_decision_tree_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Bagging Decision Trees
    response_vars = ['multinomial_YZ']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "BaggingDT", create_clf=create_bagging_dt, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Bagging Decision Trees Regressors
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "BaggingDTRegr", create_clf=create_bagging_dt_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Random Forst Classifier
    response_vars = ['multinomial_YZ']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "RF", create_clf=create_random_forest, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # Random Forst Regressor
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "RFRegr", create_clf=create_random_forest_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # AdaBoost
    response_vars = ['multinomial_YZ']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "AdaBoost", create_clf=create_adaboost, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # AdaBoostRegressor
    response_vars = ['log_returns']
    model_params = [{'n_estimators': 100},
                    {'n_estimators': 200}]
    fit_cross_validation(asset, "AdaBoostRegr", create_clf=create_adaboost_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM
    response_vars = ['binary']
    model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                    {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMBin", create_clf=create_SVM, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM
    response_vars = ['multinomial_YZ']
    model_params = [{'kernel': 'rbf', 'decision_function_shape': 'ovo', 'gamma': 'auto'},
                    {'kernel': 'sigmoid', 'decision_function_shape': 'ovo', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMMulti", create_clf=create_SVM, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # SVM Regressor
    response_vars = ['log_returns']
    model_params = [{'kernel': 'rbf', 'gamma': 'auto'}]
    fit_cross_validation(asset, "SVMRegr", create_clf=create_SVM_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # KNN
    response_vars = ['multinomial_YZ']
    model_params = [{'n_neighbors': 5},
                    {'n_neighbors': 10}]
    fit_cross_validation(asset, "KNN", create_clf=create_KNN, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # KNN Regressor
    response_vars = ['multinomial_YZ']
    model_params = [{'n_neighbors': 5},
                    {'n_neighbors': 10}]
    fit_cross_validation(asset, "KNNRegr", create_clf=create_KNN_regressor, use_scaler=False,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Binomial
    response_vars = ['binary']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)}]
    fit_cross_validation(asset, "ANNBinary", create_clf=create_MLP, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Multinomial
    response_vars = ['multinomial_YZ']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)}]
    fit_cross_validation(asset, "ANNMulti", create_clf=create_MLP, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)

    # ANN Regressor
    response_vars = ['log_returns']
    model_params = [{'solver': 'lbfgs', 'hidden_layer_sizes': (10)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (10, 3)},
                    {'solver': 'lbfgs', 'hidden_layer_sizes': (50, 10, 3)}]
    fit_cross_validation(asset, "ANNRegr", create_clf=create_MLP_regressor, use_scaler=True,
                         model_params=model_params, response_vars=response_vars, response_params=response_params,
                         input_vars=input_vars, input_params=input_params, window_sizes=window_sizes)
