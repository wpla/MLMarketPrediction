# Market prediction using machine learning and deep learning

We are using a set of classifiers, regressors together with depp LSTM network for market prediction. The following techniques are used:

 * Logistic regression (binary, multinomial)
 * Gaussian naive Bayes
 * Support vector machines
 * k-nearest neighbors
 * Decision trees
 * bagging and boosting techniques (AdaBoost)
 * artificial neural networks
 * deep LSTM networks

## Data engineering

We are using the following technical indicators to extract data features:

 * RSI
 * Stochastic K%D
 * MACD
 * CCI
 * ATR
 * ADL
 * Williams R
 * price rate of change
 * on balance volume

## Response variables

We create the following response (endogenous) variables:

 * binary classification (up/down)
 * tertiary classification (up/sideways/down)
 * multinomial classification (strong up/up/sideways/down/strong down)

The tertiary and multinomial classification is relative to the volatility where we use EWMA volatility and drift-independent volatility (Yang & Zhang 2000).

## Input

Input files are expected to be `csv` files according to the following format:

The first lines are header lines, which contain the following key: value pairs:

 * `Symbol`: stock symbol
 * `Name`: company name
 * `Exchange`: stock exchange

The following line is the header for the `csv` columns. 

The rest of the file is the market data. The file is expected to contain the following fields: `date`, `open`, `high`, `low`, `close`, `volume`.

## Usage

To amend an input file with the data from the technical indicators, run

    python gen_indicators -o <output-file> <input-file>

To run all the classifiers, regressors and deep networks on an input file, run

    python fit_models.py <input-file>
