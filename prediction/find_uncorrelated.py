import argparse
import os
import fnmatch
import itertools

from asset import Asset
from config import config, Configuration
from log import Log
from models import fit_models

# Argument parsing
parser = argparse.ArgumentParser(prog="find_uncorrelated",
                                 description="Find n top uncorrelated stocks")
parser.add_argument("-c", "--config", metavar="PATH", nargs="*", action="store", default="../config/config.ini",
                    help="Path to config file")
parser.add_argument("--loglevel", action="store", help="Log level: DEBUG, INFO, WARN, ERROR")
parser.add_argument("--logfile", action="store", metavar="PATH", help="Path to log file.")
parser.add_argument("-n", "--number", default=10, type=int, help="Number of stocks to find")
parser.add_argument("--min_data_points", default=5000, type=int, help="Minimum number of data points in file")
parser.add_argument("path", metavar="PATH", nargs='*', help="Path to stock data input files")

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
                assets[asset.exchange + "/" + asset.symbol] = asset

for (asset_name1, asset_name2) in itertools.combinations(assets.keys(), 2):
    p = abs(assets[asset_name1].data["Close"].corr(assets[asset_name2].data["Close"]))
    print("{:20s} {:20s}: {:.2f}".format(asset_name1, asset_name2, p))
    correlations[p] = (asset_name1, asset_name2)

corrs = list(correlations.keys())
corrs.sort()

print()
print("== Top {} ==".format(args.number))

for p in corrs[:args.number]:
    asset_name1, asset_name2 = correlations[p]
    print("{:20s} {:20s}: {:.2f}".format(asset_name1, asset_name2, p))

