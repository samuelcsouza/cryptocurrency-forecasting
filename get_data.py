import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime


# Busca os dados da API
load_dotenv()

ENDPOINT_BTC = os.getenv('ENDPOINT_BTC')


def convertToDF(dfJSON):
    return(pd.json_normalize(dfJSON))


def buscar_dados():
    request = requests.get(ENDPOINT_BTC)
    todo = json.loads(request.content)
    return todo['Data']['Data']


if __name__ == '__main__':
    data = buscar_dados()
    df = convertToDF(data)

    print(df.head())

    df = df[['time', 'close']]

    df.columns = ['DS', 'Y']

    df['DS'] = datetime.fromtimestamp(df['DS'])

    print(df.head())
