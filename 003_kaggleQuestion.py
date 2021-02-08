import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

st.header('What means this data?')

fpath = 'melb_data.csv'
fhand = pd.read_csv(fpath)

# y var
y = fhand['Price']

# X var
drop_y = fhand.drop('Price',axis=1)
columns = [col for col in drop_y.columns if drop_y[col].dtypes == 'object']
x_cols = fhand[columns].astype('str')
X = x_cols.drop(['Address','SellerG','Date'],axis=1)

X_labeled = X.copy()

label_encoder = LabelEncoder()
for col in X:
    X_labeled[col] = pd.DataFrame(label_encoder.fit_transform(X[col]))

# train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_labeled,y,random_state=0)

# y_val columns
# Price

# X_val columns
# Suburb
# Type
# Method
# CouncilArea
# Regionname



model_tree = DecisionTreeRegressor()
model_tree.fit(X_train,y_train)
pred = pd.DataFrame(model_tree.predict(X_val))
st.write(pred)
