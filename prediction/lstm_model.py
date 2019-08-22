import itertools
import os
from datetime import datetime

import numpy as np
from keras.callbacks import Callback
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import config
from log import Log
from models import make_data_multicol, split_data, create_input_data, make_input_col, make_response_col, \
    create_response_data

output_path_ = None


def output_path():
    global output_path_
    if output_path_ is None:
        output_path_ = os.path.join(config().output_path(), datetime.now().strftime("%Y%m%d-%H%M"))
        os.mkdir(output_path_)
    return output_path_


class TestCallback(Callback):
    def __init__(self, X, y, logfile):
        super(Callback, self).__init__()
        self.X = X
        self.y = y
        self.logfile = logfile
        self.logfile.write("epoch;loss;metric\n")
        self.epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        loss, metric = self.model.evaluate(self.X, self.y, verbose=0)
        self.logfile.write("%d;%.4f;%.4f\n" % (self.epoch, loss, metric))


def fit_LSTM_model_categorical(asset, response_var_param, input_vars, window_size, model_layers, epochs=3,
                               outfile=None):
    Log.info("Fitting categorical LSTM model %s",
             str((response_var_param, input_vars, window_size, model_layers, epochs)))

    # Create features if necessary
    input_cols = []
    for (input_var, input_param) in input_vars:
        input_col = make_input_col(input_var, input_param)
        asset = create_input_data(asset, input_var, input_param)
        input_cols.append(input_col)

    # Create response features if necessary
    response_var, response_param = response_var_param
    response_col = make_response_col(response_var, response_param)
    asset = create_response_data(asset, response_var, response_param)

    # Make response variable y and input matrix X
    y, X = make_data_multicol(asset, response_col, input_cols, window_size, config().days())

    # Convert X to numpy array
    X = np.array([x_i.values for x_i in X])

    # Convert y to categorical variable
    y = to_categorical(np.array(y) + 2, num_classes=5)

    # Split data into training and test set
    X_train, X_test, y_train, y_test = split_data(X, y)

    model_name = "LSTM_Multi_" + "".join([chr(65 + a) for a in np.random.choice(26, 5)])

    # Construct model
    model = Sequential()
    for layer in model_layers:
        model.add(LSTM(layer, return_sequences=True))
    # We end up with 5 categories
    model.add(LSTM(5, activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['categorical_accuracy'])

    directions = [-2, -1, 0, 1, 2]

    # Train model
    logfile = open(os.path.join(output_path(), model_name + ".log"), "w")
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[TestCallback(X_test, y_test, logfile)])
    print(model.summary())
    logfile.close()

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test)

    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}%".format(acc * 100))

    # Write results to file
    if outfile is not None:
        outfile.write(
            "%s;%s;%s;%s;%s;%s;%d;%d;%.4f;%.4f\n" % (asset.symbol, response_var, response_param, str(input_vars),
                                                     model_name, str(model_layers), len(X_train), len(X_test), loss,
                                                     acc))
        outfile.flush()

    return asset


def fit_LSTM_model_regression(asset, response_var_param, input_vars, window_size, model_layers, epochs=3, outfile=None):
    Log.info("Fitting regression LSTM model %s",
             str((response_var_param, input_vars, window_size, model_layers, epochs)))

    # Create input features if necessary
    input_cols = []
    for (input_var, input_param) in input_vars:
        input_col = make_input_col(input_var, input_param)
        asset = create_input_data(asset, input_var, input_param)
        input_cols.append(input_col)

    # Create response features if necessary
    response_var, response_param = response_var_param
    response_col = make_response_col(response_var, response_param)
    asset = create_response_data(asset, response_var, response_param)

    y, X = make_data_multicol(asset, response_col, input_cols, window_size, config().days())

    # Convert X to numpy array
    X = np.array([x_i.values for x_i in X])
    y = np.array(y)

    # Split data into test and training set
    X_train, X_test, y_train, y_test = split_data(X, y)

    model_name = "LSTM_Regr_" + "".join([chr(65 + a) for a in np.random.choice(26, 5)])

    # Construct model
    model = Sequential()
    for layer in model_layers:
        model.add(LSTM(layer, return_sequences=True))
    # Final layer is a real value
    model.add(LSTM(1, activation="linear"))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['mean_squared_error'])

    # Train model
    logfile = open(os.path.join(output_path(), model_name + ".log"), "w")
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[TestCallback(X_test, y_test, logfile)])
    print(model.summary())
    logfile.close()

    # Evaluate model
    loss, mse = model.evaluate(X_test, y_test)

    print("Loss: {:.2f}".format(loss))
    print("MSE: {:.2f}".format(mse))

    # Write results to file
    if outfile is not None:
        outfile.write(
            "%s;%s;%s;%s;%s;%s;%d;%d;%.5f;%.5f\n" % (asset.symbol, response_var, response_param, str(input_vars),
                                                     model_name, str(model_layers), len(X_train), len(X_test), loss,
                                                     mse))
        outfile.flush()

    return asset


def fit_LSTM_models_categorical(asset):
    # We iterate over the following parameters
    # response_vars = ["multinomial_YZ", "multinomial_EWMA"]
    response_vars = ["multinomial_YZ"]
    response_params = [10]
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
    input_params_list = [5, 10, 20]
    window_sizes = [5, 10, 15, 21]
    model_layers = [(100, 50, 10), (50, 10), (20, 10), (10,)]
    epochs_list = [20]

    # Test parameters
    #################################
    # response_vars = ["multinomial_YZ"]
    # response_params = [10]
    # input_vars_list = [('Close',),
    #                    ('EMA',),
    #                    ("EMA", "RSI")
    #                    ]
    # input_params_list = [5, 10]
    # window_sizes = [5, 10]
    # model_layers = [(20, 10), (10,)]
    # epochs_list = [3]

    # Combine input_vars and input_params
    input_vars_param_list = []
    for (var, param) in itertools.product(input_vars_list, input_params_list):
        if len(var) > 1:
            var_list = list(zip(var, [param] * len(var)))
            input_vars_param_list.append(var_list)
        else:
            input_vars_param_list.append([(var[0], param)])

    outfile = open(os.path.join(output_path(), "LSTM_Multi_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;response_var;response_param;input_var_params;model;model_layers;n_train;n_test;loss;acc\n")

    count = 0
    for (response_var, response_param, input_vars_param, window_size, model_layer, epoch) in \
            itertools.product(response_vars, response_params, input_vars_param_list, window_sizes, model_layers,
                              epochs_list):
        count += 1
        response_var_param = (response_var, response_param)
        asset = fit_LSTM_model_categorical(asset, response_var_param, input_vars_param, window_size,
                                           model_layers=model_layer, epochs=epoch, outfile=outfile)

    outfile.close()
    Log.info("%d models fitted.", count)


def fit_LSTM_model_signal(asset, response_var, input_vars, window_size, model_layers, epochs=3,
                          outfile=None):
    Log.info("Fitting categorical LSTM model %s",
             str((response_var, input_vars, window_size, model_layers, epochs)))

    # Create features if necessary
    input_cols = []
    for input_col in input_vars:
        asset = create_input_data(asset, input_col)
        input_cols.append(input_col)

    # Create response features if necessary
    response_col = make_response_col(response_var)
    asset = create_response_data(asset, response_var)

    # Make response variable y and input matrix X
    y, X = make_data_multicol(asset, response_col, input_cols, window_size, config().days())

    # Convert X to numpy array
    X = np.array([x_i.values for x_i in X])

    # Convert y to categorical variable
    y = to_categorical(np.array(y) + 1, num_classes=3)

    # Split data into training and test set
    X_val_train, X_test, y_val_train, y_test = split_data(X, y)

    X_train, X_validate, y_train, y_validate = train_test_split(X_val_train, y_val_train)

    model_name = "LSTM_Signal_" + str(int(np.random.uniform(10000, 99999)))

    if len(np.unique(np.argmax(y_validate, axis=1))) == 1 or len(np.unique(np.argmax(y_train, axis=1))) == 1:
        Log.error("Only one class in y_train or y_validate. Skip training.")
        if outfile is not None:
            outfile.write(
                "%s;%s;%s;%s;%s;%d;%d;%s;%s;%s;%s\n" % (asset.symbol, response_var, str(input_vars),
                                                        model_name, str(model_layers), len(X_train), len(X_test),
                                                        "n/a", "n/a", "n/a", "n/a"))
            outfile.flush()
        return asset

    # Construct model
    model = Sequential()
    for layer in model_layers:
        model.add(LSTM(layer, return_sequences=True))
    # We end up with 3 categories
    model.add(LSTM(3, activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['categorical_accuracy'])

    # Train model
    logfile = open(os.path.join(output_path(), model_name + ".log"), "w")
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[TestCallback(X_test, y_test, logfile)])
    # print(model.summary())
    logfile.close()

    # Evaluate model
    loss, accuracy = model.evaluate(X_validate, y_validate)

    y_predicted_class = model.predict_classes(X_validate)
    y_true_class = np.argmax(y_validate, axis=1)

    precision = precision_score(y_true_class, y_predicted_class, average="macro")
    recall = recall_score(y_true_class, y_predicted_class, average="macro")

    # print("y true:", y_true_class)
    # print("y_predicted:", y_predicted_class)
    #
    # print("Loss: {:.2f}".format(loss))
    # print("Accuracy: {:.2f}%".format(accuracy * 100))
    # print("Precision: {:.2f}%".format(precision * 100))
    # print("Recall: {:.2f}%".format(recall * 100))

    # Write results to file
    if outfile is not None:
        outfile.write(
            "%s;%s;%s;%s;%s;%d;%d;%.4f;%.4f;%.4f;%.4f\n" % (asset.symbol, response_var, str(input_vars),
                                                            model_name, str(model_layers), len(X_train), len(X_test),
                                                            loss, accuracy, precision, recall))
        outfile.flush()

    # Save model to file
    model.save(os.path.join(output_path(), model_name + ".h5"))

    return asset


def fit_LSTM_models_signals(asset):
    # We iterate over the following parameters
    # response_vars = ["multinomial_YZ", "multinomial_EWMA"]
    response_vars = ["Signal"]
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
    window_sizes = [5, 10, 15, 21]
    model_layers = [(100, 50, 10), (50, 10), (20, 10), (10,)]
    epochs_list = [20]

    # Test parameters
    #################################
    response_vars = ["Signal_50",
                     "Signal_100",
                     "Signal_200",
                     "Signal_300"]
    input_vars = ['EMA_50',
                  'RSI_14',
                  'MACD',
                  'MACD_Signal'
                  ]
    window_sizes = [5, 14, 28]
    model_layers = [(20, 10), (10,)]
    epochs_list = [3]

    # Create all combinations of input_vars
    input_vars_list = []
    for i in range(1, len(input_vars) + 1):
        for j in itertools.combinations(input_vars, r=i):
            input_vars_list.append(j)

    outfile = open(os.path.join(output_path(), "LSTM_Signal_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;response_var;input_var_params;model;model_layers;n_train;"
                  "n_test;loss;acc;precision;recall\n")

    count = 0
    for (response_var, input_vars, window_size, model_layer, epoch) in \
            itertools.product(response_vars, input_vars_list, window_sizes, model_layers,
                              epochs_list):
        count += 1
        asset = fit_LSTM_model_signal(asset, response_var, input_vars, window_size,
                                      model_layers=model_layer, epochs=epoch, outfile=outfile)

    outfile.close()
    Log.info("%d models fitted.", count)


def fit_LSTM_models_regression(asset):
    # We iterate over the following parameters
    response_vars = ["log_returns"]
    response_params = [0]
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
    input_params_list = [5, 10, 20]
    window_sizes = [5, 10, 15, 21]
    model_layers = [(100, 50, 10), (50, 10), (20, 10), (10,)]
    epochs_list = [20]

    # Test parameters
    #################################
    # response_vars = ["log_returns"]
    # response_params = [0]
    # input_vars_list = [('Close',),
    #                    ('EMA',),
    #                    ("EMA", "RSI")
    #                    ]
    # input_params_list = [5, 10]
    # window_sizes = [5, 10]
    # model_layers = [(20, 10), (10,)]
    # epochs_list = [3]

    # Combine input_vars and input_params
    input_vars_param_list = []
    for (var, param) in itertools.product(input_vars_list, input_params_list):
        if len(var) > 1:
            var_list = list(zip(var, [param] * len(var)))
            input_vars_param_list.append(var_list)
        else:
            input_vars_param_list.append([(var[0], param)])

    outfile = open(os.path.join(output_path(), "LSTM_Regr_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;response_var;response_param;input_var_params;model;model_layers;n_train;n_test;loss;mse\n")

    for (response_var, response_param, input_vars_param, window_size, model_layer, epoch) in \
            itertools.product(response_vars, response_params, input_vars_param_list, window_sizes, model_layers,
                              epochs_list):
        response_var_param = (response_var, response_param)
        asset = fit_LSTM_model_regression(asset, response_var_param, input_vars_param, window_size,
                                          model_layers=model_layer, epochs=epoch, outfile=outfile)

    outfile.close()


def fit_LSTM_models(asset):
    # fit_LSTM_models_categorical(asset)
    # fit_LSTM_models_regression(asset)

    fit_LSTM_models_signals(asset)
