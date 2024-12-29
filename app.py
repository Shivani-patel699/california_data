import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st


#load the dataset
cal_data=fetch_california_housing()
df=pd.DataFrame(cal_data.data,columns=cal_data.feature_names)
df['MedHouseVal']=cal_data.target
df.head()

#title of application
st.title('california housing price prediction for xyz brokerage')
