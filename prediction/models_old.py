from pycm import ConfusionMatrix
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from log import Log
from models import make_data


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


