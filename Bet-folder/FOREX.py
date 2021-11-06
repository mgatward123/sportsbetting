
#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr


yf.pdr_override() 

import math  
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import statistics
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans

#import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
from sklearn.externals import joblib
import ta


#import xgboost
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
from sklearn import  metrics, model_selection
#from xgboost.sklearn import XGBClassifier


#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr
from scipy.signal import argrelextrema


yf.pdr_override() 

import math  
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import statistics
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans

#import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
from sklearn.externals import joblib
import ta


#import xgboost
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (12,8)
from sklearn import  metrics, model_selection
#from xgboost.sklearn import XGBClassifier

# DO THE REST OF JAN HAVE TO DELETE ROW


#client = MongoClient()
#database = client['okcoindb']
#collection = database['historical_data']

# Retrieve price, v_ask, and v_bid data points from the database.

import pandas as pd
import yfinance as yf
import time
from pandas_datareader import data as pdr


yf.pdr_override() 

import math  
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import statistics
import numpy as np
from numpy.linalg import norm
from sklearn import linear_model
from sklearn.cluster import KMeans

#import statsmodels.api as sm
from scipy import stats
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import scipy
import datetime
import json
import seaborn as sns
from sklearn.externals import joblib
import ta



ticker = [
# S AND P 500        WHEN ORDERING ADD TO 9
'AUDCAD=X',     
'AUDCHF=X',     
'AUDJPY=X',     
'EURAUD=X', 
'EURGBP=X',
'EURJPY=X',     
'EURUSD=X',     
'EURCHF=X',     
'GBPUSD=X',      
'GBPEUR=X',      
'GBPNZD=X',
'GBPJPY=X',     
'GBPCAD=X', 
'NZDCAD=X',     
'GBPCHF=X', 
'NZDJPY=X',     
'NZDUSD=X',  
'NZDEUR=X',     
'USDCAD=X',     
'USDCHF=X',   
'USDJPY=X',     
'CADJPY=X',     
'CADCHF=X',    
'EURSGD=X', 
'GBPSGD=X',     
'EURNZD=X',     
'NZDJPY=X',     
'EURCAD=X',
"CADJPY=X",     


]


def get_best_hmm_model(X, max_states, max_iter = 10000):
    best_score = -(10 ** 10)
    best_state = 0
    
    for state in range(1, max_states + 1):
        hmm_model = GaussianHMM(n_components = state, random_state = 100,
                                covariance_type = "diag", n_iter = max_iter).fit(X)
        if hmm_model.score(X) > best_score:
            best_score = hmm_model.score(X)
            best_state = state
    
    best_model = GaussianHMM(n_components = best_state, random_state = 100,
                                covariance_type = "diag", n_iter = max_iter).fit(X)
    return best_model

# Normalized st. deviation
def std_normalized(vals):
    return np.std(vals) / np.mean(vals)

# Ratio of diff between last price and mean value to last price
def ma_ratio(vals):
    return (vals[-1] - np.mean(vals)) / vals[-1]

# z-score for volumes and price
def values_deviation(vals):
    return (vals[-1] - np.mean(vals)) / np.std(vals)

# General plots of hidden states
def plot_hidden_states(hmm_model, data, X, column_price):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(hmm_model.n_components, 3, figsize = (15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    hidden_states = model.predict(X)
    
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = hidden_states == i
        ax[0].plot(data.index, data[column_price], c = 'grey')
        ax[0].plot(data.index[mask], data[column_price][mask], '.', c = colour)
        ax[0].set_title("{0}th hidden state".format(i))
        ax[0].grid(True)
        
        ax[1].hist(data["future_return"][mask], bins = 30)
        ax[1].set_xlim([-0.1, 0.1])
        ax[1].set_title("future return distrbution at {0}th hidden state".format(i))
        ax[1].grid(True)
        
        ax[2].plot(data["future_return"][mask].cumsum(), c = colour)
        ax[2].set_title("cummulative future return at {0}th hidden state".format(i))
        ax[2].grid(True)
        
    plt.tight_layout()


def mean_confidence_interval(vals, confidence):
    a = 1.0 * np.array(vals)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m, m + h

def compare_hidden_states(hmm_model, cols_features, conf_interval, iters = 1000):
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(len(cols_features), hmm_model.n_components, figsize = (15, 15))
    colours = cm.prism(np.linspace(0, 1, hmm_model.n_components))
    
    for i in range(0, model.n_components):
        mc_df = pd.DataFrame()
    
        # Samples generation
        for j in range(0, iters):
            row = np.transpose(hmm_model._generate_sample_from_state(i))
            mc_df = mc_df.append(pd.DataFrame(row).T)
        mc_df.columns = cols_features
    
        for k in range(0, len(mc_df.columns)):
            axs[k][i].hist(mc_df[cols_features[k]], color = colours[i])
            axs[k][i].set_title(cols_features[k] + " (state " + str(i) + "): " + str(np.round(mean_confidence_interval(mc_df[cols_features[k]], conf_interval), 3)))
            axs[k][i].grid(True)
            
    plt.tight_layout()








for x in ticker:
    print(x)

    #datayahoo = pdr.get_data_yahoo(x,  period = "7d",  interval = "1m")
    datayahoo = pdr.get_data_yahoo(x, interval = "1d", start="1950-07-15", end="2022-12-20")

    datayahoo = datayahoo.reset_index()
 
    #datayahoo = datayahoo[:-5]

    #print(datayahoo) 


    datayahooopen = datayahoo['Open'].values.tolist()
    datayahoohigh = datayahoo['High'].values.tolist() 
    datayahoolow = datayahoo['Low'].values.tolist()
    datayahooclose = datayahoo['Close'].values.tolist()
    datayahoovolume = datayahoo['Volume'].values.tolist()

    data = pd.DataFrame(columns= ['Open', 'High', 'Low', 'Close', 'Volume'])


    #data['Timestamp']  =  bdate + datayahoodate
    data['Open'] =  datayahooopen
    data['High'] = datayahoohigh
    data['Low'] =  datayahoolow
    data['Close'] =  datayahooclose
    data['Volume'] =  datayahoovolume
        
        
    price = data['Close']

    #data['Timestamp'] =  pd.to_datetime(data['Timestamp'])
    data = data.drop_duplicates()


    price = price * 100000

    #price = price[:-24] 

        

    max_idx = list(argrelextrema(price.values, np.greater, order=1)[0])
    min_idx = list(argrelextrema(price.values, np.less, order=1)[0]) 

    idx = max_idx + min_idx


    idx.sort()

    current_idx = idx + [len(price.values) - 1] 

    current_pat = price.values[current_idx]

    start = min(current_idx)
    end = max(current_idx)

    list_of_values = current_pat.tolist()

    df = pd.DataFrame({'Price':list_of_values})

    df1 = df

    try:
        model = get_best_hmm_model(X = df1, max_states = 10, max_iter = 1000000)
    except:
        print('means weight')

    print("Best model with {0} states ".format(str(model.n_components)))  
        

    predictionDataset = df.iloc[-1:]

    prediction = model.predict(predictionDataset)

    # LOGIC SHORT WHEN 0; NO POSITION WHEN 1; LONG WHEN 2;
    print(x)
    #print(len(prediction))
    print(sum(prediction))

