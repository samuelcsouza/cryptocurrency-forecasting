import json
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from pytictoc import TicToc


def get_data(coin, sample_data=True):

    # Verificação dos parâmetros #
    if coin.upper() not in ('BTC', 'ETH', 'XRP'):
        err_msg = coin + ' is a invalid coin!'
        raise ValueError(err_msg)

    # Variáveis de Ambiente #
    # Nome da variável de Ambiente
    name_coin = "_SAMPLE_DATA" if sample_data else "_ALL_DATA"
    name_coin = coin.upper() + name_coin

    print("\nGetting ", "sample" if sample_data else "all",
        " data from ", coin.upper(), " coin ...")

    # Carre as variáveis de ambiente
    load_dotenv()

    # Tempo de processamento
    t = TicToc()
    t.tic()

    # Busca pela variável da moeda selecionada
    coin_url = os.getenv(name_coin)

    # Request na API #
    # Realiza a Request
    request = requests.get(coin_url)
    # Pega os dados vindos da Request
    data = json.loads(request.content)
    # Pega o campo "Data"
    content = data.get("Data")
    # Pega novamente o segundo campo "Data"
    content = content.get("Data")

    print("Dataset has been loaded! Formatting ...")

    # Manipulação dos dados #
    # Primeira observação
    df = pd.json_normalize(content[0])

    # Para cada linha da lista
    for i in range(1, len(content)):
        # Pega a observação (dicionário)
        observation = content[i]  

        # Transforma em um DataFrame
        df_temp = pd.json_normalize(observation)

        # Concatena com o DataFrame inicial
        df = pd.DataFrame.append(df, df_temp)

    t.toc("Done!")
    # Retorna o DataFrame com as observações
    return df
