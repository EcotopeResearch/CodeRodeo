# http://logging.r-forge.r-project.org/sample_session.php
# based around the standard Python logging library
# install.packages("logging")
library(logging)
logReset()
basicConfig()  # Defualt INFO

setwd("~/GitHub/CodeRodeo/Ecology/")

getLogger()[['level']]
getLogger()[['handlers']]

loginfo('does it work?')
logwarn('my %s is %d', 'name', 5)


basic_log <- function(){
  logerror("Hey this is an ERROR level log")
  logwarn("Hey this is an WARNING level log")
  loginfo("Hey this is an INFO level log")
  logdebug("Hey this is an DEBUG level log")
  logfine("Hey this is an FINE level log, it's below DEBUG!")
  logfiner("Hey this is an FINER level log, it's below DEBUG!")
  logfinest("Hey this is an FINEST level log, it's below DEBUG!")
  
}
basic_log()

# Set level by handler
setLevel("FINER", getHandler('basic.stdout'))
basic_log()
# But remember the logger is still filtering before the handlers!
setLevel("FINEST")
basic_log()
# Is this a bug with finest level?

##########################################################################
# Logging to file
addHandler(writeToFile, file="testing.log", level='DEBUG')
getLogger()[['handlers']]  # 2 handlers now!
# removeHandler('basic.stdout')  # Remove by name
basic_log()

##########################################################################
# Formatters
defaultFormat <- function(record) {
  text <- paste(record$timestamp, paste(record$levelname, record$logger, record$msg, sep=':'))
}
# could write custom formatters...



##########################################################################
# Maybe we should use logger here...
# https://daroczig.github.io/logger/index.html
##########################################################################

library(logger)