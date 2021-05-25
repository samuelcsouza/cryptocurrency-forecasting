base::suppressMessages({
  library(dplyr)
  library(httr)
  library(jsonlite)
  library(prophet)
  library(tictoc)
  library(future)
})

source('functions/get_data.R')

btc <- get_data("btc", FALSE)

# Forecasting
df <- btc %>% 
  dplyr::select('ds' = date, 'y' = close)

