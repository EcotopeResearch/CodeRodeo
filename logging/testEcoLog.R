# Need our own logger for server operations using base R
# Meant to be stepped through since errors will terminate a source

setwd("C:/Users/paul/Documents/GitHub/CodeRodeo/CodeRodeo/logging/")
source("ecoLog.R")

# logFine, logDebug, logInfo, logWarn, logError
basic_log <- function(){
  logError("Hey this is an ERROR level log")
  logWarn("Hey this is an WARNING level log")
  logInfo("Hey this is an INFO level log")
  logDebug("Hey this is an DEBUG level log")
  logFine("Hey this is an FINE level log, it's below DEBUG!")
}

# Test with out capturing errors. ################################
fileName = "testNoErrors.log"
openLog(fileName, append = FALSE, captureErrors = FALSE)

basic_log()

mean()
adsfdsaf(42)

paste(Sys.time(),'last message')

# Close up the logger
closeLog()


# Set up logger test with errors  ################################
fileName = "testWithErrors.log"
openLog(fileName, append = FALSE, captureErrors = TRUE)

basic_log()

mean()
adsfdsaf(42)

paste(Sys.time(),'last message')

# Close up the logger
closeLog()

# Test with append  #############################################
fileName = "testAppend.log"
openLog(fileName, append = TRUE, captureErrors = FALSE, logLevel="FINE")

basic_log()

mean()
adsfdsaf(42)

paste(Sys.time(),'last message')

# Close up the logger
closeLog()
