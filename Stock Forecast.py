import numpy as np
import pandas as pd
import quandl, math
import datetime
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

#Getting dataset from quandl (can only be used when dataconnection is ON)
quandl.ApiConfig.api_key = 'xiPzkgdGa25bWY7xpgUg'
df = quandl.get('WIKI/GOOGL')

#Eliminating the not required columns
df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#Adding Columns of percentage changes
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/(df['Adj. Low']*100)
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open'])/(df['Adj. Open']*100)

df=df[['Adj. Close','HL_PCT','PCT_CHNG','Adj. Volume']]

forecast_col = 'Adj. Close'
#Prediction for next 15 days
forecast_out = int(15)

df['label']=df[forecast_col].shift(-forecast_out)


X= np.array(df.drop(['label'],1))

X=preprocessing.scale(X)


#deciding training and testing data
X_forecast_out= X[-forecast_out:]
X= X[:-forecast_out]

y=np.array(df['label'])
y=y[:-forecast_out]


X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#Applying Algorithm,fitting ,predicting
clf=LinearRegression()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

forecast_prediction = clf.predict(X_forecast_out)

df.dropna(inplace=True)
df['forecast']=np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix= last_unix+one_day

for i in forecast_prediction:
    next_date= datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)]+[i]


#Plotting the forecast
df['Adj. Close'].plot(figsize=(50,80),color="green")
df['forecast'].plot(figsize=(50,80),color="orange")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
