import numpy as np
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def fit_LSTM_models_test(asset):
    fit_LSTM_model_test_multi()
    fit_LSTM_model_test_regr()


def fit_LSTM_model_test_multi():
    model_name = "LSTM"
    model_parameters = "(5, 5)"
    model = Sequential()
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(5, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    directions = [-2, -1, 0, 1, 2]

    data = [[[[1, 1], [2, 2], [3, 3]], -1],
            [[[3, 3], [2, 2], [1, 1]], 2],
            [[[4, 4], [5, 5], [7, 7]], 1],
            [[[1, 1], [-1, -1], [10, 10]], -2]]

    X = []
    y_ = []
    for i in np.random.choice(len(data), 10):
        X.append(data[i][0])
        y_.append(data[i][1])

    X = np.array(X)
    y_ = np.array(y_)
    y = to_categorical(y_ + 2, num_classes=5)

    print(X[:3])
    print(y[:3])
    # return

    model.fit(X, y, epochs=3, validation_split=0.2)
    print(model.summary())

    loss, acc = model.evaluate(X, y)

    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}%".format(acc * 100))


def fit_LSTM_model_test_regr():
    model_name = "LSTM"
    model_parameters = "(5, 5)"
    model = Sequential()
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(1, activation="linear"))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    directions = [-2, -1, 0, 1, 2]

    data = [[[[1, 1], [2, 2], [3, 3]], 0.55],
            [[[3, 3], [2, 2], [1, 1]], 0.88],
            [[[4, 4], [5, 5], [7, 7]], -100.44],
            [[[1, 1], [-1, -1], [10, 10]], 3.6666]]

    X = []
    y = []
    for i in np.random.choice(len(data), 10000):
        X.append(data[i][0])
        y.append(data[i][1])

    X = np.array(X)
    y = np.array(y)

    print(X[:3])
    print(y[:3])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    print(model.summary())

    loss, mse = model.evaluate(X_test, y_test)

    print("Loss: {:.2f}".format(loss))
    print("MSE: {:.2f}".format(mse))

    sample = 2
    X_pred = np.array([data[sample][0]])
    y_exp = data[sample][1]
    y_pred = model.predict(X_pred)

    print(y_exp, y_pred)
