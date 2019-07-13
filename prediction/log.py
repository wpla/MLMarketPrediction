import logging
import os


class Log:
    instance = None

    @staticmethod
    def init_instance(logfile=None, loglevel=logging.DEBUG, logformat="%(asctime)s - %(levelname)s - %(message)s"):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
            Log.instance.setLevel(loglevel)
            formatter = logging.Formatter(logformat)
            # Filehandler
            if logfile is not None:
                (path, file) = os.path.split(logfile)
                if not os.path.exists(path):
                    os.makedirs(path)
                file_handler = logging.FileHandler(logfile)
                file_handler.setLevel(loglevel)
                file_handler.setFormatter(formatter)
                Log.instance.addHandler(file_handler)
            # Console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(loglevel)
            console_handler.setFormatter(formatter)
            Log.instance.addHandler(console_handler)
        else:
            raise Exception("Logger already initialized")

    @staticmethod
    def get_logger():
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        return Log.instance

    @staticmethod
    def debug(str, *args, **kwargs):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        Log.instance.debug(str, *args, **kwargs)

    @staticmethod
    def info(str, *args, **kwargs):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        Log.instance.info(str, *args, **kwargs)

    @staticmethod
    def warn(str, *args, **kwargs):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        Log.instance.warning(str, *args, **kwargs)

    @staticmethod
    def error(str, *args, **kwargs):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        Log.instance.error(str, *args, **kwargs)

    @staticmethod
    def fatal(str, *args, **kwargs):
        if Log.instance is None:
            Log.instance = logging.getLogger("app")
        Log.instance.fatal(str, *args, **kwargs)
