import argparse

from asset import Asset
from config import config, Configuration
from log import Log
from models import gen_indicators

# Argument parsing
parser = argparse.ArgumentParser(prog="gen_indicators",
                                 description="Generate various indicators for asset data")
parser.add_argument("-c", "--config", metavar="PATH", nargs="*", action="store", default="../config/config.ini",
                    help="Path to config file")
parser.add_argument("--loglevel", action="store", help="Log level: DEBUG, INFO, WARN, ERROR")
parser.add_argument("--logfile", action="store", metavar="PATH", help="Path to log file.")
parser.add_argument("-o", "--output", metavar="FILE", required=True, action="store", help="Output file")
parser.add_argument("filename", metavar="FILE", help="files to process")

args = parser.parse_args()

# configuration files
Configuration.init_instance(args.config, args)

# configure logger
Log.init_instance(config().logfile(), config().loglevel(), config().logformat())

if __name__ == "__main__":
    Log.info("Generating indicators for %s", args.filename)
    asset = Asset()
    asset.read_csv(args.filename)
    asset = gen_indicators(asset)
    asset.write_csv(args.output)
    Log.info("Output written to %s", args.output)
