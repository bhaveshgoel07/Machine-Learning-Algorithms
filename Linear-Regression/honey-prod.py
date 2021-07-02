
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("honeyproduction.csv")
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year['year'].values.reshape(-1,1)
y = prod_per_year['totalprod'].values

plt.scatter(X, y)


lireg = linear_model.LinearRegression().fit(X,y)

y_predict = lireg.predict(X)
plt.plot(X, y_predict)
x_future = np.array(range(2013,2050)).reshape(-1, 1)
y_predict = lireg.predict(x_future)
plt.plot(x_future, y_predict)
plt.show()