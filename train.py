import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


df = pd.read_csv('hiring.csv')


df.rename(columns={'test_score(out of 10)':'test_score'},inplace=True)

df.experience.fillna(0, inplace = True)
df.test_score.fillna(df.test_score.mean(),inplace= True)

exp_num = {0:0 , 'one': 1 , 'two': 2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11}
df['experience']= df['experience'].map(exp_num)

x= df.drop(columns=['salary($)'])
y= df['salary($)']

regressor = LinearRegression()
regressor.fit(x,y)

joblib.dump(regressor,'deploy_test.pkl')

##print(regressor.predict([[10,7,7]]))

