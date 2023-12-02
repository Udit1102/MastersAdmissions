This project aims to predict the chances of getting admission into masters program and graph a Decision Tree with the help of DecisionTreeRegression by using following dependent variables:
1. GRE Scores ( out of 340 )
2. TOEFL Scores ( out of 120 )
3. University Rating ( out of 5 )
4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
5. Undergraduate GPA ( out of 10 )
6. Research Experience ( either 0 or 1 )
Dependent variable: Chance of Admit ( ranging from 0 to 1 )

#Working
1. Loading the data from a csv file.
2. Understanding and explloring the data and getting the initial glimpses.
3. Getting the data ready for further use by machine learning model.
4. Splitting the data into training and testing set and fitting the model.
5. Tuning the model for better accuracy using max_depth parameter.
6. Again improving the accuracy by pruning (tuning ccp_alpha).
7. Finally generating the Decision Tree and making the predictions for the sample data.

#Output
1. Accuracy
2. Graphs max_depth vs accuracy and ccp_alpha vs accuracy.
3. Final graph and text based Decision Tree along with predictions.