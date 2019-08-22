import argparse
import os
import fnmatch
import itertools
import sys
import shutil
sys.path.append("../prediction")

from asset import Asset
from config import config, Configuration
from log import Log

# Argument parsing
parser = argparse.ArgumentParser(prog="find_uncorrelated",
                                 description="Find n top uncorrelated stocks")
parser.add_argument("-c", "--config", metavar="PATH", nargs="*", action="store", default="../config/config.ini",
                    help="Path to config file")
parser.add_argument("--loglevel", action="store", help="Log level: DEBUG, INFO, WARN, ERROR")
parser.add_argument("--logfile", action="store", metavar="PATH", help="Path to log file.")
parser.add_argument("-n", "--number", default=10, type=int, help="Number of stocks to find")
parser.add_argument("--min_data_points", default=5000, type=int, help="Minimum number of data points in file")
parser.add_argument("--dest_dir", "-d", action="store", required=True, help="Destination directory.")
parser.add_argument("path", metavar="PATH", nargs='*', help="Paths to stock data input files")

args = parser.parse_args()

# configuration files
Configuration.init_instance(args.config, args)

# configure logger
Log.init_instance(config().logfile(), config().loglevel(), config().logformat())

assets = {}
correlations = {}

if __name__ == "__main__":
    for path in config().input_path():
        Log.info("Searching in %s", path)
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename, "*.csv"):
                Log.info("File %s", filename)
                asset = Asset()
                asset.read_csv(os.path.join(path, filename))
                if len(asset.data["Volume"][asset.data["Volume"] > 10000]) < args.min_data_points:
                    Log.info("Ignoring data set since it has too few entries ({} instead of {})".format(
                        len(asset.data["Volume"][asset.data["Volume"] > 10000]), args.min_data_points))
                    continue
                if asset.exchange is None or asset.symbol is None:
                    Log.info("Ignoring data set since exchange or symbol is missing")
                    continue
                if asset.data["Close"].iloc[0] / asset.data["Close"].iloc[-1] > 1000:
                    Log.info("Ignoring data since ratio between first price and last price > 1000.")
                shutil.copyfile(os.path.join(path, filename), os.path.join(args.dest_dir, filename))

