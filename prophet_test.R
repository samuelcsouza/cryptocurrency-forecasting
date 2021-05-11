
request <- httr::GET(url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&allData=true&api_key%20=dc41e2121dafd2413debfaf774366c736fd0390de2b711ee2a7ce00ae24eb444", 
                     encode = "json")

request_content = httr::content(request, "text")

request_json <- jsonlite::fromJSON(request_content)
df <- request_json$Data$Data

remove(request, request_content, request_json)

library(dplyr)

data <- df %>% 
  dplyr::mutate(ds = as.Date(as.POSIXct(time, origin="1970-01-01"))) %>% 
  dplyr::select(ds, y = close)

library(prophet)

m <- prophet(data)

# 1
future <- prophet::make_future_dataframe(m, periods = 1)
tail(future)

# 2
forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

prophet::prophet_plot_components(m, forecast)

plot(m, forecast)

