
import pandas as pd
import numpy as np
import pickle
import sklearn

df = pd.read_csv("cleaned_data.csv")
df.dropna(inplace=True)
df.drop(["Project Code"], axis=1, inplace=True)

P_group = {
    'HRDT' : 0, 'ARV' : 1, 'ACT': 2, 'MRDT': 3, 'ANTM': 4
}

df["Product Group"] = df["Product Group"].replace(P_group)
df.drop(["PO Sent to Vendor Date", "Item Description", "Molecule/Test Type", "PQ First Sent to Client Date", "Managed By"], axis=1, inplace=True)

y_temp = df["Vendor"]
final_y =  pd.factorize(y_temp)[0]+1

country = df["Country"]
final_country = pd.factorize(country)[0]+1
df["Country"].replace(final_country, inplace=True)

df["new_country"] = final_country
df["new_vendors"] = final_y

df.drop(["Line Item Insurance (USD)"], inplace=True, axis =1)
df = df.drop(['Weight (Kilograms)', 'Freight Cost (USD)'], axis=1)
df = df.drop(['PQ #', 'PO / SO #', 'ASN/DN #'], axis=1)
df['Fulfill Via'] = df['Fulfill Via'].replace({'Direct Drop': 0, 'From RDC': 1})
df['First Line Designation'] = df['First Line Designation'].replace({'No': 0, 'Yes': 1})


x = df.drop(["new_vendors", "Brand", "Dosage", "Dosage Form", "Manufacturing Site", "Country", "ID", "Scheduled Delivery Date", "Delivered to Client Date", "Delivery Recorded Date", "Sub Classification", "Vendor", "Line Item Value"], axis=1)
y = df["new_vendors"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# prediction = model.predict(X_test)
# prediction

pickle.dump(model.open('nodal.pkl', 'wb'))