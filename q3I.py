import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

# Alphavantage only allows up to 5 queries per minute for a free user, hence the 4 stock symbols in MY_SYMBOL_LIST
# I have written this code to work for any number of stocks, so if you want, remove the ]#  after "TCS" in MY_SYMBOL_LIST
# (and change the api_key to a premium one if you have one)
# and see it work for 10 stocks, but that would take a lot of time.

MY_SYMBOL_LIST=["RELIANCE", "HDFC", "ADANIPORTS", "TCS", "INFY"]#, "TITAN", "HINDUNILVR", "BAJFINANCE", "ADANITRANS", "ICICIBANK" ]

# p-value significance limit

SIG=0.2

#Sometimes getting stock data takes a lot of time. 

def get_historical_data(symbol):
    api_key = "U399LYM9LZEUX2M0"
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}.BSE&apikey={api_key}&outputsize=compact'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
    df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': "close", '5. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df=df["close"].to_frame(symbol)
    print(".")
    return df

def get_close_df(stock_list):
    df=pd.DataFrame()
    temp=[]
    for i in stock_list:
        temp.append(get_historical_data(i))
    df=pd.concat(temp, axis=1, join='inner')
    return df

def get_ols_pair_pvalue(close_df, symbol1, symbol2):
    x = close_df[symbol1].tolist()
    y = close_df[symbol2].tolist()
    
    x = sm.add_constant(x)

    result = sm.OLS(y, x).fit()
    residuals=result.resid
    pvalue=ts.adfuller(residuals)[1]
    return({f"{symbol1} and {symbol2}" : pvalue})

if __name__=="__main__":
    c_df=get_close_df(MY_SYMBOL_LIST)
    pval_dict={}
    for i in MY_SYMBOL_LIST:
        for j in MY_SYMBOL_LIST:
            if(i!=j):
                pval_dict.update(get_ols_pair_pvalue(c_df, i, j))

    for w in sorted(pval_dict, key=pval_dict.get, reverse=False):
        if(pval_dict[w]<SIG):
            print(w, pval_dict[w])
    



