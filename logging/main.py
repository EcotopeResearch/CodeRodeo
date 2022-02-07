"""
Python Logging Lesson

Why log?
- Easier for tracing issues in large projects.
- If designing applications can debug non-standard cases.
- Easy to adjust logging level (i.e. debugging messages vs error messages)
- Searchable unlike standard print

Python logger:
https://docs.python.org/3/library/logging.html

The basic classes defined by the module, together with their functions, are listed below.

Loggers: expose the interface that application code directly uses.
Handlers: send the log records (created by loggers) to the appropriate destination.
Filters: provide a finer grained facility for determining which log records to output.
Formatters: specify the layout of log records in the final output.

Good Tutorials outside of docs:
https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

Overview (open in ingonito):
https://towardsdatascience.com/8-advanced-python-logging-features-that-you-shouldnt-miss-a68a5ef1b62d


Extending the logger:

Formating Special Attributes:
https://docs.python.org/3/library/logging.html#logrecord-attributes

CLI:
https://docs.python.org/3/howto/logging-cookbook.html#a-cli-application-starter-template

Rotating File Handlers: Since sometimes logs get huge:
https://docs.python.org/3/howto/logging-cookbook.html#using-file-rotation

Multiple Handlers
https://docs.python.org/3/howto/logging-cookbook.html#multiple-handlers-and-formatters
https://stackoverflow.com/questions/7274732/extending-the-python-logger

Logger Adapters:
https://docs.python.org/3/library/logging.html?highlight=loggeradapter#logging.LoggerAdapter
https://stackoverflow.com/questions/59837559/how-to-modify-python-loggeradapter

Fun ideas
    - Class to gather all tic toc items and log them at exit. Tictoc for timing could be logger.tic() logger.toc()
        At exit:
        https://docs.python.org/3/library/atexit.html#atexit-example

    -v/ Make color prints in the terminal. Remember that color prints package you made before.

    -v/ Make a standard Ecotope config file for the logger, or a bonus package ("Ecolog-y", Elogger?)

    -v/ Catching uncaught exceptions!
        https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
"""

import os
import logging

os.chdir(os.path.dirname(__file__))

file = 'CodeRodeo.log'
# logging.basicConfig(filename=file,
#                     #level=logging.INFO,
#                     level=logging.DEBUG,
#                     format='%(asctime)s | %(levelname)s | %(message)s ',
#                     datefmt='%m/%d/%Y %I:%M:%S %p'
#                     #filemode='w'
#                     )

"""
A good convention to use when naming loggers is to use a module-level logger, in each module which uses logging, named as follows:

logger = logging.getLogger(__name__)
This means that logger names track the package/module hierarchy, and it's intuitively obvious where events are logged just from the logger name.
"""
logging.basicConfig(filename=file,
                    filemode='w',
                    level=logging.INFO,
                    # level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s ',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.info("Here's the logger: %s" % (__name__))


def testLog():
    logger.debug("Hey this is an DEBUG level log")
    logger.info("Hey this is an INFO level log")
    logger.warning("Hey this is an WARNING level log")
    logger.error("Hey this is an ERROR level log")
    logger.critical("Hey this is an CRITICAL level log")
    raise Exception("Raising Exceptions for fun!")


testBasicLog()








# Logging with configs and user defined objects:
# https://docs.python.org/3/library/logging.config.html#user-defined-objects

import Elogger
logger = Elogger.get_logger(__name__, file,
                            loggerLevel="DEBUG",
                            fileLevel="DEBUG",
                            streamLevel="INFO")

testBasicLog()
