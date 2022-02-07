# Elogger.py
# could also build from log file with custom functions:
# https://docs.python.org/3/library/logging.config.html#import-resolution-and-custom-importers

import logging
import sys


_log_format = "%(asctime)s | %(name)s | %(levelname)8s | %(message)s"
_logger = logging.getLogger(__name__)


class ColorFormatter(logging.Formatter):
    """
    Logging colored formatter, adapted from
    https://stackoverflow.com/a/56944256/3638629
    """

    green = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.blue + self.fmt + self.reset,
            logging.INFO: self.green + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def string_to_level(txt):
    if txt.upper() == "DEBUG":
        return logging.DEBUG
    elif txt.upper() == "INFO":
        return logging.INFO
    elif txt.upper() == "WARNING":
        return logging.WARNING
    elif txt.upper() == "ERROR":
        return logging.ERROR
    else:
        raise Exception("Unsupported logging level")


# https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    _logger.critical("Uncaught exception:", exc_info=(exc_type,
                                                      exc_value,
                                                      exc_traceback
                                                      )
                     )


def get_file_handler(file, level, writeMode):
    file_handler = logging.FileHandler(file, mode=writeMode)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler


def get_stream_handler(level):
    stream_handler = logging.StreamHandler()  # by default, sys.stderr
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ColorFormatter(_log_format))
    return stream_handler


def get_logger(name, file="log.log",
               loggerLevel="INFO",
               fileLevel="INFO",
               streamLevel="INFO",
               catchUncaught=True,
               writeMode='a'):
    _logger = logging.getLogger(name)

    # Interestingly the logger level supercedes the handler logging levels,
    # which all default to logging.WARNING
    # https://stackoverflow.com/questions/57429455/logging-module-streamhandler-does-not-seem-to-attach-to-logger
    _logger.setLevel(string_to_level(loggerLevel))
    # Add both handlers to the logger
    _logger.addHandler(get_file_handler(file, fileLevel, writeMode))
    _logger.addHandler(get_stream_handler(streamLevel))

    if catchUncaught:
        sys.excepthook = handle_exception

    return _logger
