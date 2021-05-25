
get_data <- function(coin, sample_data = TRUE){
  
  # Verificação dos parâmetros
  if(!toupper(coin) %in% c('BTC', 'ETH', 'XRP'))
    stop(coin, 'is a invalid coin!')
  
  # Variáveis de ambiente
  name_coin <- dplyr::if_else(sample_data, "_SAMPLE_DATA", "_ALL_DATA")
  name_coin <- paste0(toupper(coin), name_coin)
  
  message(
    "Getting ", dplyr::if_else(sample_data, "sample", "all"),
    " data from ", toupper(coin), " coin ..."
  )
  
  tictoc::tic()
  
  coin_url <- Sys.getenv(name_coin)
  
  # Requisição
  request <- httr::GET(url = coin_url,
                       encode = "json")
  
  request_content = httr::content(request, "text")
  
  request_json <- jsonlite::fromJSON(request_content)
  df <- request_json$Data$Data
  
  remove(request, request_content, request_json)
  
  data <- df %>% 
    dplyr::mutate(date = as.Date(as.POSIXct(time, origin="1970-01-01")))
  
  tictoc::toc()
  
  return(data)
  
}