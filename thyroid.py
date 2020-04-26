  
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Precision , Recall
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score

dataset = loadtxt('annthyroid.csv' , delimiter = ',')
train_size = int((7200*(0.7)))
X_test = dataset[0:train_size,0:6]
Y_test = dataset[0:train_size,6]
X_val = dataset[train_size:,0:6]
Y_val = dataset[train_size:,6]


import pandas as pd

from google.colab import files
file = files.upload()
X_train = pd.read_csv("xtrain.csv", header=None)
Y_train = pd.read_csv("ytrain.csv", header=None)
X_test = pd.read_csv("xtest.csv", header=None)
Y_test = pd.read_csv("ytest.csv", header=None)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() 

classifier.add(Dense(units = 32, activation = 'relu', input_dim = 6))
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, Y_train, batch_size = 1, epochs = 10)

Y_pred = classifier.predict(X_test) # predictions on the test set
Y_predi = [ 1 if y>=0.5 else 0 for y in Y_pred ] # how to get sigmoid? ? ?
Y_pred_train = classifier.predict(X_train) # predictions on the training set 

# output on screen
total=0
correct=0
wrong=0
for i in Y_predi:
  total=total+1
  if (Y_test.at[i,0] == Y_predi[i]): # is this the correct way? can I get sigmoid??
    correct=correct+1
  else:
    wrong=wrong+1

a=correct/total
b=a*100

print('Total ' + str(total))
print('Correct Predics ' + str(correct))
print('Incorrect Predics ' + str(wrong))
print('Percentage ' + str(b))
print('---------------------------')

_, accuracy = classifier.evaluate(X_test, Y_test)
print('Accuracy on test data: %.2f' % (accuracy*100))

_, accuracytrain = classifier.evaluate(X_train, Y_train)
print('Accuracy on training data: %.2f' % (accuracytrain*100))

import seaborn as sns
sns.set()
plt.hist(Y_pred_train, bins=10, color='red') #histogram on training set predictions
plt.ylim(0,6480)
plt.title('Predictions on training data')
plt.xlabel('Anomaly Score (x 100)')
plt.ylabel('Total')

'''
import matplotlib.pyplot as plt
plt.hist(Y_pred, bins=10) # histogram on testing set predicitons
plt.ylim(0,720)
plt.title('Predictions on testing data')
plt.xlabel('Anomaly Score (x 100)')
plt.ylabel('Total')
'''

