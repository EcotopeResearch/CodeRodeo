# Need our own logger for server operations using base R

# This will use camelCase

library(crayon)
# logging with sink
# ?sink

# logFine, logDebug, logInfo, logWarn, logError
# openLog, and always closeLog!

# Logging setup ###########################################
zz <<- NULL # File conn
ll <<- NULL # Log level

openLog <- function(fileName, append=TRUE, captureErrors = FALSE, logLevel = "INFO") {
  ll <<- logLevel
  
  # Capture messages and errors to a file
  zz <<- file(fileName, open = ifelse(append, "a", "wt"))
  sink(zz, type="output", append=append, split=TRUE)
  
  # Due to limitations in we can't split the messages so error messages can only
  # go to one place.  
  if(captureErrors) sink(zz, type="message", append=append)
}


# Reset sink, close connection
closeLog <- function() {
  # Close up the sink and log files
  if(sink.number(type="message") > 0) {
    sink(type="message")
  }
  i <- sink.number()
  while (i > 0) {
    sink()
    i <- i - 1
  }
  
  close(zz) # close the error log connection
  zz <<- NULL # blank from mem
}


# Logging functions  ###############################
logLevels <<- list(ERROR = 10, WARN = 20, INFO = 30, DEBUG = 40, FINE = 50)

# logFine, logDebug, logInfo, logWarn, logError
logFine <- function (msg) {
  formatedMessage(paste0("FINE ::", msg), blue, "FINE")
}
logDebug <- function (msg) {
  formatedMessage(paste0("DEBUG ::", msg), green, "DEBUG")
}
logInfo <- function (msg) {
  formatedMessage(paste0("INFO ::", msg), white, "INFO")
}
logWarn <- function (msg) {
  formatedMessage(paste0("WARNING::", msg), yellow, "WARN")
}
logError <- function (msg, stop1 = FALSE) {
  if(stop1) stop(formatedMessage(paste0("ERROR ::", msg), red, "ERROR"))
  formatedMessage(paste0("ERROR ::", msg), red, "ERROR")
}


# Messaging ###############################
formatedMessage <- function(msg, colorFct, level) {
  if(logLevels[[ll]] >= logLevels[[level]]) {
    cat(colorFct(paste(Sys.time(), Sys.timezone(), msg, "\n")))
  }
}

