import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("df.csv")

x = pd.read_csv("X.csv")
y = pd.read_csv("Y.csv")

y.drop("Unnamed: 0", axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)



pickle.dump(model,open('model.pkl','wb'))