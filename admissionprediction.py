import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#loading the data and exploring it

df = pd.read_csv("Admission_Predict.csv")
#print(df.info())
#print(df.head(2))
df.columns = df.columns.str.replace(" ", "_").str.lower()
#print(df.columns)
x = df.loc[:, "gre_score":"research"]
#for i in x.columns:
#	print(i,x[i].value_counts(),"\n")

y = df.iloc[:,-1]

#splitting the data and training the model

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#fitting the model

dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)
print(dtree.score(x_test, y_test))

#tuning the model for max_depth

depths = range(1,21)
accuracy_depth = []
for i in depths:
    dtree_depth = DecisionTreeRegressor(max_depth=i)
    dtree_depth.fit(x_train, y_train)
#    print(f"accuracy at max_depth {i} is {dtree_depth.score(x_test, y_test)}")
    accuracy_depth.append(dtree_depth.score(x_test, y_test))

#plotting depths vs accuracy

plt.plot(depths, accuracy_depth)
plt.xlabel("depths")
plt.ylabel("accuracy")
plt.title("depths vs accuracy")
#plt.show()

best_depth = depths[np.argmax(accuracy_depth)]
print(f"the best accuracy at max_depth {best_depth} is {max(accuracy_depth)}")

#tuning the model for ccp_alpha at max_depth = 1

ccp = np.logspace(-3,1,20)
accuracy_ccp = []
for i in ccp:
    dtree_ccp = DecisionTreeRegressor(max_depth=best_depth, ccp_alpha=i)
    dtree_ccp.fit(x_train, y_train)
    acc = dtree_ccp.score(x_test, y_test)
    accuracy_ccp.append(acc)
#    print(f"the max accuracy at ccp {i} is {acc}")

#plotting ccp vs accuracy

plt.plot(ccp, accuracy_ccp)
plt.xlabel("ccp_alpha")
plt.ylabel("accuracy")
plt.title("ccp_alpha vs accuracy")
#plt.show()
best_ccp = ccp[np.argmax(accuracy_ccp)]
print(f"the best accuracy at ccp {best_ccp} is {max(accuracy_ccp)}")

##no major improvement in accuracy with ccp_alpha

#making the final model and predicting with the model

dtree_final = DecisionTreeRegressor(max_depth=best_depth, ccp_alpha=best_ccp)
dtree_final.fit(x_train, y_train)
y_pred = dtree_final.predict(x_test)
print(dtree_final.score(x_test, y_test))

#graphing the tree

plt.figure(figsize=(10,10))
tree.plot_tree(dtree_final, feature_names=x.columns, filled=True)
print(tree.export_text(dtree_final, feature_names=x.columns.tolist()))
#plt.show()

#sample input data
sample = np.array([[320,100,4,4,4,7,1]])
print(f"chances of getting admitted is {dtree_final.predict(sample)*100}")