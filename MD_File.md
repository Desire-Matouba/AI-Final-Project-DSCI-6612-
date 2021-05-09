## MACHINE LEARNING FOR MEDICAL DIAGNOSIS

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


```python

```
