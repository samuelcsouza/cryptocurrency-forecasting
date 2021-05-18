base::suppressMessages({
  library(dplyr)
  library(httr)
  library(jsonlite)
  library(prophet)
  library(tictoc)
})

source('functions/get_data.R')

btc_sample_data <- get_data("btc")
