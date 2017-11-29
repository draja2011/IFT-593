# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:14:40 2017

@author: Sweet Dee
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import norm

df = pd.read_csv('TestBalaData.csv')

df.head()

print(df.corr().iloc[:,0])

X = df.loc[:,['Ambient Temperature ', 'Irradiance ']].as_matrix()
y = df.loc[:,['Cell Temperature']].as_matrix()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the training sets
y_pred = regr.predict(X_test)

# Prediction error
err = y_test - y_pred

# r2 score
print('-------------Results-----------------------')
print('R2: {0:.2f}'.format(r2_score(y_test, y_pred)))


# Scatter plot of error
plt.plot(err,'bo')
plt.xlabel('Sample')
plt.ylabel('Error')
plt.show()

# Plot Histogram of Residual
(mu, sigma) = norm.fit(err)
n, bins, patches = plt.hist(err, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Histogram of Error')
plt.grid(True)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=2)
plt.show()

# Plot qqplot
res = stats.probplot(np.reshape(err,len(err)), plot=plt)
plt.title('QQplot of Error')

# Plot VOC vs Ambient for actual and predicted values
fig = plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(X_test[:,0], y_test.squeeze(), 'bo', label='actual')
plt.plot(X_test[:,0], y_pred.squeeze(), 'ro', label='predicted')
plt.xlabel('Ambient Temperature')
plt.ylabel('Cell Temperature')
plt.grid(True)
plt.legend(loc=3)

# Plot VOC vs Irradiance for actual and predicted values
plt.subplot(122)
plt.plot(X_test[:,1], y_test.squeeze(), 'bo', label='actual')
plt.plot(X_test[:,1], y_pred.squeeze(), 'ro', label='predicted')
plt.xlabel('Irradiance')
plt.ylabel('Cell Temperature')
plt.grid(True)
plt.legend(loc=3)
plt.show()