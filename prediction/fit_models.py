import argparse

from asset import Asset
from config import config, Configuration
from log import Log
from lstm_model import fit_LSTM_models
from models import fit_models, fit_models_crossvalidated, fit_models_crossvalidated_test

# Argument parsing
parser = argparse.ArgumentParser(prog="run_models",
                                 description="Runs various stock prediction models")
parser.add_argument("-c", "--config", metavar="PATH", nargs="*", action="store", default="../config/config.ini",
                    help="Path to config file")
parser.add_argument("--loglevel", action="store", help="Log level: DEBUG, INFO, WARN, ERROR")
parser.add_argument("--logfile", action="store", metavar="PATH", help="Path to log file.")
parser.add_argument("--output_path", action="store", default=".", help="Where to put output files")
parser.add_argument("--days", default=5000, type=int, help="Fit models using the last n days")
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
        # fit_models(asset)
        # fit_models_crossvalidated(asset)
        # fit_models_crossvalidated_test(asset)
        fit_LSTM_models(asset)
