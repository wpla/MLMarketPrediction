import argparse

from asset import Asset
from config import config, Configuration
from log import Log
import lstm_model
import models

# Argument parsing
parser = argparse.ArgumentParser(prog="run_models",
                                 description="Runs various stock prediction models")
parser.add_argument("-c", "--config", metavar="PATH", nargs="*", action="store", default="../config/config.ini",
                    help="Path to config file")
parser.add_argument("--loglevel", action="store", help="Log level: DEBUG, INFO, WARN, ERROR")
parser.add_argument("--logfile", action="store", metavar="PATH", help="Path to log file.")
parser.add_argument("--output_path", action="store", default=".", help="Where to put output files")
parser.add_argument("--days", default=5000, type=int, help="Fit models using the last n days")
parser.add_argument("--models", default="clf,reg,lstm",
                    help="Models to fit: clf=classifiers, reg=regressors, lstm=deep LSTM network")
parser.add_argument("--clf", default="logreg,nb,dt,bdt,rf,adaboost,svn,knn,nn",
                    help="Classifier to fit: logreg=logistic regression classifier, "
                         "nb=Gaussian naive Bayes, "
                         "dt=decision trees, "
                         "bdt=baggging decision trees, "
                         "rf=random forest classifier, "
                         "adaboost=AdaBoost, "
                         "svm=SVM classifier, "
                         "knn=KNN classifier, "
                         "nn=neural network classifier")
parser.add_argument("--reg", default="linreg,dt,bdt,rf,adaboost,svm,knn,nn",
                    help="Regressors to fit: linreg=linear regression, "
                         "dt=decision tree regressor,"
                         "bdt=bagging decision trees regressor, "
                         "rf=random forest regressor, "
                         "adaboost=AdaBoost regressor, "
                         "svm=SVM regressor, "
                         "knn=KNN regressor, "
                         "nn=neural network regressor")
parser.add_argument("filename", metavar="FILE", nargs='*', help="Stock data input files to process")

args = parser.parse_args()

# configuration files
Configuration.init_instance(args.config, args)

# configure logger
Log.init_instance(config().logfile(), config().loglevel(), config().logformat())

if __name__ == "__main__":
    for filename in config().input_filenames():
        Log.info("Running models for %s", filename)
        asset = Asset()
        asset.read_csv(filename)
        for model in args.models.split(","):
            if model == "lstm":
                lstm_model.fit_LSTM_models(asset)
            elif model == "clf":
                models.fit_classifiers(asset, classifiers=args.clf.split(","))
            elif model == "reg":
                models.fit_regressors(asset, regressors=args.reg.split(","))
