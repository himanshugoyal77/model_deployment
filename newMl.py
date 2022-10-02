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

# print(X_test.iloc[9])
#
#
# temp = [[7102.00, 1, 1, 1, 60.00, 100, 7.50, 8.12, 1, 2]]
#
# print("after ppt")
# print(model.predict(temp))

# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

pickle.dump(model,open('model.pkl','wb'))