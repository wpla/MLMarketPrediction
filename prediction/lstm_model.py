from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import os

from config import config


def fit_LSTM_model(asset):
    model_name = "LSTM"
    model_parameters = "(5, 5)"
    model = Sequential()
    model.add(LSTM(5, return_sequences=True))
    model.add(LSTM(5, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    directions = [-2, -1, 0, 1, 2]

    data = [[[[1], [2], [3]], -1],
            [[[3], [2], [1]], 2],
            [[[4], [5], [7]], 1],
            [[[1], [-1], [10]], -2]]

    X = []
    y_ = []
    for i in np.random.choice(len(data), 10000):
        X.append(data[i][0])
        y_.append(data[i][1])

    X = np.array(X)
    y_ = np.array(y_)
    y = to_categorical(y_ + 2, num_classes=5)

    model.fit(X, y,
              epochs=3,
              validation_data=(X, y))
    print(model.summary())

    loss, acc = model.evaluate(X, y)

    print("Loss: {:.2f}".format(loss))
    print("Accuracy: {:.2f}%".format(acc*100))

    outfile = open(os.path.join(config().output_path(), "LSTM_" + asset.symbol + ".csv"), "w")
    outfile.write("asset;model;model_params;loss;acc\n")
    outfile.write("%s;%s;%s;%.2f;%.2f\n" % (asset.symbol, model_name, model_parameters, loss, acc))
    outfile.close()

