### Project Title

Heart Disease Prediction

### Presentation

The short video presentation for this project is at https://www.youtube.com/watch?v=OgwYvf08PgE


The slide presentation used in this video is https://github.com/Desire-Matouba/Heart-Disease-Prediction_ML/blob/main/Heart%20Disease%20Prediction_ML.pdf

### Project Description 

This analysis is about predicting the heart disease condition of patients. There are 2 conditions: absence (value 0) to presence (values 1) of heart disease. The European Cardiology Society has found that machine learning  model is more than 90% accurate in analyzing variables to determine a person's risk of suffering a heart attack or death in the future while human prediction are less efficient.

We get the data from UCI Machine Learning Repository. Below is the dataset link:
http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

The dataset has information about 303 patients. Out of the 76 features available, we used only 14 of them for our study. Some of the features are the following:
cp = chest pain
trestbps = resting blood pressure
chol - cholesterol
fbs = fasting blood sugar      
restecg = resting electrocardiographic results
thalach = maximum heart rate achieved
exang = exercise induced angina

We provide some plots to visualize the data such as the heatmap that shows the correlation among the features, and histogram for the data distribution of each feature of the dataset.

We divide the data into training and testing using 70% for the training data and 30% for the testing data. 

We provide the accuracy, precision, ROC curve and confusion matrix for the following machine learning models:
Logistic Regression
Decision Tree  
Random Forest
Xgboost

Furthermore, we also provide accuracy and precision of Neural Network. 
After comparing the models above, we add two custom ensemble: the first ensemble has 2 leayers and uses the random forest and logistic regression. The second ensemble uses the decision tree and xgoost with 2 layers. We add the superlearners to improve the accuracy of the models.

### Objective

Machine learning allows building models to quickly analyze data and deliver results. Machine Learning help historical and help healthcare Service providers to make better decisions on Patient’s disease diagnosis.

By analyzing the data, we will be able to Predict the accuracy of occurrence of the disease. In our project. This intelligent system for disease Prediction plays a major role in controlling the Disease and maintaining the good health status of People by predicting accurate disease risk.


### Feature Selection and Splitting Dataset

We did feature selection and we divided the dataset into training and testing. The class column from the dataset  is dependent variable and the features columns are independent variables.

#feature selection

X = df.drop(columns=['class'])

Y = df['class']

print("Features Extraction Sucessfull")


### Machine Learning Models 

We imported the model, trained the model, predicted the model, and found the accuracy for each model. We used the following models to predict the heart disease condition of the patients in the dataset:

Logistic Regression

Decision Tree

Random Forest

Xgboost

Neural Network

### Logistic Regression 

Logistic Regression Accuracy is 0.93 and Precision is 0.947.

Logistic Regression predicted that 48 patients without heart
disease are correctly predicted as not having heart disease and 36
patients with heart disease are correctly predicted as having heart
disease.

It also incorrectly predicted that 2 patients who do not have heart
disease are predicted as having heart disease (false positive) and 4
patients who have heart disease are predicted as not having the
heart disease (false negative).

### Decision Tree 

Decision Tree Accuracy is 0.76 and Precision is 0.705.
Decision Tree predicted that 37 patients without heart disease are correctly
predicted as not having heart disease and 31 patients with heart disease are
correctly predicted as having heart disease.

It also incorrectly predicted that 13 patients who do not have heart disease
are predicted as having heart disease (false positive) and 9 patients who have
heart disease are predicted as not having the heart disease (false negative).

#Model Improvement to increase accuracy

5-fold Cross Validation and Bagging

New Accuracy : 0.8

### Random Forest

Random Forest Predicted the data with higher accuracy than Decision Tree. 

Random Forest Accuracy is 0.87 and Precision is 0.83.

Random Forest predicted that 43 patients without heart disease are correctly
predicted as not having heart disease and 35 patients with heart disease are
correctly predicted as having heart disease.

It also incorrectly predicted that 7 patients who do not have heart disease are
predicted as having heart disease (false positive) and 5 patients who have heart
disease are predicted as not having the heart disease (false negative).

### XGBoost 

XGBoost Predicted the data with higher accuracy than Decision Tree. 

XGBoost Accuracy is 0.81 and Precision is 0.81.

XGBoost predicted that 43 patients without heart disease are correctly predicted
as not having heart disease and 30 patients with heart disease are correctly
predicted as having heart disease.

It also incorrectly predicted that 7 patients who do not have heart disease are
predicted as having heart disease (false positive) and 10 patients who have heart
disease are predicted as not having the heart disease (false negative).

### Neural Network 

NN Predicted the data with higher accuracy than Decision Tree. 

NN Train Accuracy is 0.88.

NN Test Accuracy is 0.79.

### End Results 

From the results of the above models, Logistic Regression has highest accuracy of 0.93 and highest precision of 0.947. Furthermore, Logistic Regression has the highest Area Under Curve (AUC) of 0.95. This project concludes that Logistic Regression is the best model to predict heart disease condition.

### References 

https://www.healthline.com/health/heart-disease/statistics#Who-is-at-risk?

https://www.who.int/en/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

https://www.hcplive.com/view/machine-learning-boasts-90-accuracy-rate-for-predicting-heart-attack-death


```python

```
