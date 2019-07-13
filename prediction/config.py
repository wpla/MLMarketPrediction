import configparser
import logging


class Configuration:
    instance = None

    def __init__(self, filenames, args=None):
        self.config = configparser.ConfigParser(default_section="General", interpolation=None)
        self.config.read(filenames)
        self.filenames = filenames
        self.args = args

    def ini_files(self):
        return self.filenames

    def input_filenames(self):
        if hasattr(self.args, "filename") and len(self.args.filename) > 0:
            return self.args.filename
        return [x.strip() for x in self.config.get("General", "filenames").split(",")]

    def input_path(self):
        if hasattr(self.args, "path"):
            return self.args.path
        return [x.strip() for x in self.config.get("General", "path").split(",")]

    def output_path(self):
        if hasattr(self.args, "output_path"):
            return self.args.output_path
        return self.config.get("General", "output_path")

    def days(self):
        if hasattr(self.args, "days"):
            return self.args.days
        return self.config.getint("General", "days", fallback=5000)

    def loglevel(self):
        if hasattr(self.args, "loglevel") and self.args.loglevel is not None:
            str_value = self.args.loglevel
        else:
            str_value = self.config.get("Log", "loglevel")
        if str_value == "DEBUG":
            return logging.DEBUG
        elif str_value == "INFO":
            return logging.INFO
        elif str_value == "WARN":
            return logging.WARN
        elif str_value == "ERROR":
            return logging.ERROR
        elif str_value == "FATAL":
            return logging.FATAL

    def logfile(self):
        if hasattr(self.args, "logfile") and self.args.logfile is not None:
            return self.args.logfile
        return self.config.get("Log", "logfile")

    def logformat(self):
        if self.config.has_option("Log", "logformat"):
            return self.config.get("Log", "logformat")
        return "%(asctime)s - %(levelname)s - %(message)s"

    @staticmethod
    def init_instance(filenames, args=None):
        Configuration.instance = Configuration(filenames, args)

    @staticmethod
    def get_instance():
        if Configuration.instance is None:
            raise Exception("Static instance of Configuration class not initialized")
        return Configuration.instance


def config():
    return Configuration.get_instance()
