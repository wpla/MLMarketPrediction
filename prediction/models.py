from pycm import ConfusionMatrix
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, \
    AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import indicators
import itertools
import os
import numpy as np
import pandas as pd

from config import config
from log import Log


def gen_indicators(asset):
    # Generate EMA indicator
    Log.info("Generating EMA...")
    EMA_5 = indicators.gen_EMA(asset.data["Close"], n=5)
    EMA_10 = indicators.gen_EMA(asset.data["Close"], n=10)
    EMA_20 = indicators.gen_EMA(asset.data["Close"], n=20)
    EMA_50 = indicators.gen_EMA(asset.data["Close"], n=50)
    asset.append("EMA_5", EMA_5)
    asset.append("EMA_10", EMA_10)
    asset.append("EMA_20", EMA_20)
    asset.append("EMA_50", EMA_50)

    # Generate RSI indicator
    Log.info("Generating RSI...")
    RSI_10 = indicators.gen_RSI(asset.data["Close"], n=10)
    RSI_20 = indicators.gen_RSI(asset.data["Close"], n=20)
    RSI_50 = indicators.gen_RSI(asset.data["Close"], n=50)
    asset.append("RSI_10", RSI_10)
    asset.append("RSI_20", RSI_20)
    asset.append("RSI_50", RSI_50)

    # Generate Stochastics K%D indicator
    Log.info("Generating Stochastics K%D...")
    Stochastics_K_14, Stochastics_D_fast, Stochastics_D_slow = \
        indicators.gen_Stochastics(asset.data["Close"], K_n=14)
    asset.append("Stochastics_K_14", Stochastics_K_14)
    asset.append("Stochastics_D_fast", Stochastics_D_fast)
    asset.append("Stochastics_D_slow", Stochastics_D_slow)

    # Generate MACD indicator
    Log.info("Generating MACD...")
    MACD, MACD_Signal = indicators.gen_MACD(asset.data["Close"])
    asset.append("MACD(12,26,9)", MACD)
    asset.append("MACD_Signal(12,26,9)", MACD_Signal)

    # Generate CCI indicator
    Log.info("Generating CCI...")
    CCI_10 = indicators.gen_CCI(asset.data, n=10)
    CCI_20 = indicators.gen_CCI(asset.data, n=20)
    CCI_50 = indicators.gen_CCI(asset.data, n=50)
    asset.append("CCI_10", CCI_10)
    asset.append("CCI_20", CCI_20)
    asset.append("CCI_50", CCI_50)

    # Generate ATR indicator
    Log.info("Generating ATR...")
    ATR_10 = indicators.gen_ATR(asset.data, n=10)
    ATR_20 = indicators.gen_ATR(asset.data, n=20)
    ATR_50 = indicators.gen_ATR(asset.data, n=50)
    asset.append("ATR_10", ATR_10)
    asset.append("ATR_20", ATR_20)
    asset.append("ATR_50", ATR_50)

    # Generate ADL indicator
    Log.info("Generating ADL...")
    ADL = indicators.gen_ADL(asset.data)
    asset.append("ADL", ADL)

    # Generate returns
    Log.info("Generating returns...")
    returns, log_returns, ann_log_returns, mon_log_returns, qu_log_return, yearly_log_returns = \
        indicators.gen_returns(asset.data)

    asset.append("returns", returns)
    asset.append("log_returns", log_returns)
    asset.append("ann_log_returns", ann_log_returns)
    asset.append("monthly_log_returns", mon_log_returns)
    asset.append("quarterly_log_returns", qu_log_return)
    asset.append("yearly_log_returns", yearly_log_returns)

    # Generate simple volatility
    Log.info("Generating simple volatility...")
    vola_10, ann_vola_10 = indicators.gen_SimpleVola(asset.data["Close"], days=10)
    vola_20, ann_vola_20 = indicators.gen_SimpleVola(asset.data["Close"], days=20)
    asset.append("vola_10", vola_10)
    asset.append("ann_vola_10", ann_vola_10)
    asset.append("vola_20", vola_20)
    asset.append("ann_vola_20", ann_vola_20)

    # Generate EWMA volatility
    Log.info("Generating EWMA volatility...")
    EWMA_ann_vola_10 = indicators.gen_EWMA_Vola(asset.data["Close"], n=10)
    EWMA_ann_vola_20 = indicators.gen_EWMA_Vola(asset.data["Close"], n=20)
    asset.append("EWMA_ann_vola_10", EWMA_ann_vola_10)
    asset.append("EWMA_ann_vola_20", EWMA_ann_vola_20)

    # Generate Yang & Zhang volatility
    Log.info("Generating Yang & Zhang volatility...")
    YZ_vola_10 = indicators.gen_YZ_Vola(asset.data, days=10)
    YZ_vola_20 = indicators.gen_YZ_Vola(asset.data, days=20)
    asset.append("YZ_Vola_10", YZ_vola_10)
    asset.append("YZ_Vola_20", YZ_vola_20)

    # Generate binary and multinomial response variables
    Log.info("Generating response variables...")
    binary = indicators.gen_binary_response(asset.data, ann_log_returns)
    multinomial_YZ_10 = indicators.gen_multinomial_response(asset.data, ann_log_returns, YZ_vola_10)
    multinomial_EWMA_10 = indicators.gen_multinomial_response(asset.data, ann_log_returns, EWMA_ann_vola_10)
    multinomial_YZ_20 = indicators.gen_multinomial_response(asset.data, ann_log_returns, YZ_vola_20)
    multinomial_EWMA_20 = indicators.gen_multinomial_response(asset.data, ann_log_returns, EWMA_ann_vola_20)

    asset.append("binary", binary)
    asset.append("multinomial_YZ_10", multinomial_YZ_10)
    asset.append("multinomial_YZ_20", multinomial_YZ_20)
    asset.append("multinomial_EWMA_10", multinomial_EWMA_10)
    asset.append("multinomial_EWMA_20", multinomial_EWMA_20)

    return asset


def make_data(asset, response_col, input_col, window_size=10, days=5000):
    y = []  # endogenous variable
    X = []  # Matrix of exogenous variables

    response_var = asset.data[response_col]
    input_vars = asset.data[input_col]
    input_len = len(asset.data[input_col])

    for t in range(input_len - 1, input_len - days, -(window_size + 1)):
        y_t = response_var[t]
        X_t = input_vars[t - window_size:t]

        if not X_t.isna().any() and not pd.isna(y_t):
            y.append(y_t)
            X.append(X_t)

    return y, X


def to_str(d):
    def to_str_(k, v):
        if v is None:
            return "%s: n/a" % (k)
        elif type(v) == str:
            return "%s: %s" % (k, v)
        else:
            return "%s: %.2f" % (k, v)

    return ", ".join([to_str_(k, v) for k, v in d.items()])


def print_confusion_scores(y_test, y_pred):
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)
    Log.info("TPR (True positive): %s", to_str(cm.TPR))
    Log.info("TNR (True negative): %s", to_str(cm.TNR))
    Log.info("PPV (Positive predictive value): %s", to_str(cm.PPV))
    Log.info("NPV (Negative predictive value): %s", to_str(cm.NPV))
    Log.info("ACC (Accuracy): %s", to_str(cm.ACC))
    Log.info("AUC (Area under the ROC curve): %s", to_str(cm.AUC))


def add_result(results, method_name, y_col, X_col, window_size, score):
    if method_name not in results:
        results[method_name] = {}
    if X_col not in results[method_name]:
        results[method_name][X_col] = {}
    if y_col not in results[method_name][X_col]:
        results[method_name][X_col][y_col] = {}
    if window_size not in results[method_name][X_col][y_col]:
        results[method_name][X_col][y_col][window_size] = {}
    results[method_name][X_col][y_col][window_size]["score"] = score
    return results


def fit_binary_logistic_regression(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Binary logistic regression")

    clf = LogisticRegression(penalty='l2', solver='lbfgs')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    results = add_result(results, "Binomial logistic regression", y_col, X_col, window_size, score)
    return results


def fit_multinomial_logistic_regression(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Multinomial logistic regression")

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    results = add_result(results, "Multinomial logistic regression", y_col, X_col, window_size, score)
    return results


def fit_naive_bayes(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Gaussian Naive Bayes")

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results = add_result(results, "Gaussian Naive Bayes", y_col, X_col, window_size, score)
    return results


def fit_multinomial_naive_bayes(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Multinomial Naive Bayes")

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results = add_result(results, "Multinomial Naive Bayes", y_col, X_col, window_size, score)
    return results


def fit_support_vector_machines(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Support Vector Machines")

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results = add_result(results, "Support Vector Machines", y_col, X_col, window_size, score)
    return results


def fit_KNN(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** KNN")

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    results = add_result(results, "KNN", y_col, X_col, window_size, score)
    return results


def fit_decision_trees(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Decision Trees")

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    results = add_result(results, "Decision Trees", y_col, X_col, window_size, score)
    return results


def fit_adaboost(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** AdaBoost")

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    results = add_result(results, "AdaBoost", y_col, X_col, window_size, score)
    return results


def fit_bagging_logistic_regression(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** Bagging (Logistic Regression)")

    clf = BaggingClassifier(LogisticRegression(solver='lbfgs', multi_class='multinomial'),
                            n_estimators=5, max_samples=0.5, max_features=0.5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results = add_result(results, "Bagging (log. regression)", y_col, X_col, window_size, score)
    return results


def fit_ANN(asset, y_col, X_col, window_size, results):
    y, X = make_data(asset, response_col=y_col, input_col=X_col, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    Log.info("*** ANN")

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 2), random_state=1)
    clf.fit(X_train_scaled, y_train)
    score = clf.score(X_test_scaled, y_test)

    results = add_result(results, "ANN(6,2)", y_col, X_col, window_size, score)
    return results


def make_response_col(response_var, response_param):
    if response_param is None:
        return response_var
    elif response_var in ['multinomial_YZ', 'multinomial_EWMA']:
        return response_var + "_" + str(response_param)
    return response_var


def create_response_data(asset, response_var, response_param):
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


def create_input_data(asset, input_var, input_param):
    input_col = make_input_col(input_var, input_param)

    if input_col not in asset.data:
        Log.info("Need to make input variable: %s", input_col)
        if input_var == "RSI" and input_param is not None:
            RSI = indicators.gen_RSI(asset.data["Close"], input_param)
            asset.append(input_col, RSI)
            Log.info("%s created", input_col)
        elif input_var == "Stochastic" and input_param is not None:
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


def fit_cross_validation(asset,
                         model_name,
                         create_clf=None,
                         model_params=None,
                         use_scaler=False,
                         response_vars=None,
                         response_params=None,
                         input_vars=None,
                         input_params=None,
                         window_sizes=None):
    Log.info("== Fitting model %s for %s", model_name, asset.symbol)

    model_scores = {}

    if asset.symbol is None:
        asset.symbol = "_UNK_"

    outfile = open(os.path.join(config().output_path(), model_name + "_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;model;response_var;response_param;input_var;input_param;window_size;model_param;"
                  "cv_score_mean;cv_score_std;test_score\n")

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
                y, X = make_data(asset, response_col=response_col, input_col=input_col, window_size=window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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
                outfile.write("%s;%s Training;%s;%s;%s;%s;%s;%s;%.4f;%.4f;%.4f\n" % (asset.symbol, model_name,
                                                                                     response_var, response_param,
                                                                                     input_var,
                                                                                     input_param, str(window_size),
                                                                                     str(model_param),
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
            asset.symbol, model_name, response_var, response_param, input_var, input_param, str(window_size), model_param_str,
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
    input_vars = ["Close", "returns", "log_returns", "ann_log_returns",
                  "RSI", "Stochastic", "MACD", "CCI", "ATR", "ADL"]
    input_params = {"RSI": [5, 10, 20, 50],
                    "Stochastic": [5, 10, 20, 50],
                    "MACD": [5, 10, 20, 50],
                    "CCI": [5, 10, 20, 50],
                    "ATR": [5, 10, 20, 50]
                    }
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
