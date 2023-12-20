import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def prediction(gre,toefl,rating,sop,lor,gpa,research):
	df = pd.read_csv("Admission_Predict.csv")
	df.columns = df.columns.str.replace(" ", "_").str.lower()
	x = df.loc[:, "gre_score":"research"]
	y = df.iloc[:,-1]
#normalization
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
#splitting the data and training the model
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
#fitting the model
	dtree = DecisionTreeRegressor(max_depth=4)
	dtree.fit(x_train, y_train)
	sample = [[gre,toefl, rating,sop,lor,gpa,research]]
	normalized_sample = scaler.transform(sample)
	pred = dtree.predict(normalized_sample)
	return pred

