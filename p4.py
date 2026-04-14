#two hidden layers for performing classification and predicting the class.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
X,Y=make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)
scalar=MinMaxScaler()
scalar.fit(X)
X=scalar.transform(X)
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X,Y,epochs=500)
Xnew,Yreal=make_blobs(n_samples=3,centers=2,n_features=2,random_state=1)
Xnew=scalar.transform(Xnew)
Ynew=model.predict_classes(Xnew)
for i in range(len(Xnew)):
 print("X=%s,Predicted=%s,Desired=%s"%(Xnew[i],Ynew[i],Yreal[i]))


#two hidden layers for performing classification and predicting the probability of class.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
X,Y=make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)
scalar=MinMaxScaler()
scalar.fit(X)
X=scalar.transform(X)
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X,Y,epochs=500)
Xnew,Yreal=make_blobs(n_samples=3,centers=2,n_features=2,random_state=1)
Xnew=scalar.transform(Xnew)
Yclass=model.predict_classes(Xnew)
Ynew=model.predict_proba(Xnew)
for i in range(len(Xnew)):
 print("X=%s,Predicted_probability=%s,Predicted_class=%s"%(Xnew[i],Ynew[i],Yclass[i]))


#two hidden layers for performing linear regression and predicting values.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalerX, scalerY = MinMaxScaler(), MinMaxScaler()
X = scalerX.fit_transform(X)
Y = scalerY.fit_transform(Y.reshape(-1, 1))
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(loss='mse', optimizer='adam')
model.fit(X, Y, epochs=1000, verbose=0)
Xnew, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
Xnew = scalerX.transform(Xnew)
Ynew = model.predict(Xnew)
Ynew = scalerY.inverse_transform(Ynew)
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], Ynew[i][0]))



