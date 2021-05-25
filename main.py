from get_data import *

if __name__ == '__main__':

    btc_sample_data = get_data(coin='btc', sample_data=True)

    print(btc_sample_data.head())
    

