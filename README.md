# Credit Risk Prediction using Machine Learning

This project focuses on predicting credit default risk using machine learning models. The goal is to build a reliable system that can identify high risk customers while handling class imbalance and optimizing decision thresholds.

Problem Statement

Financial institutions need accurate models to identify customers who are likely to default on loans. The dataset is imbalanced and requires careful handling to improve detection of default cases.

Approach

Data preprocessing and feature preparation  
Handling class imbalance using SMOTE  
Model training using Logistic Regression Random Forest and XGBoost  
Model evaluation using ROC AUC and Precision Recall AUC  
Threshold optimization to control recall and precision trade off  
Cross validation for model stability  
Hyperparameter tuning using GridSearchCV  

Results

Logistic Regression ROC AUC 0.73  
Random Forest ROC AUC 0.75  
XGBoost ROC AUC 0.78  

Cross validation mean ROC AUC 0.7815 with low variance  

Default recall improved from around 0.30 to up to 0.74 after imbalance handling and threshold tuning  

Best XGBoost parameters learning rate 0.1 max depth 3 n estimators 100  

Key Insights

Handling class imbalance significantly improves detection of defaulters  
Threshold tuning allows control over risk sensitivity  
XGBoost provides the best balance between precision and recall  
Model performance is stable across different data splits  

Technologies Used

Python  
Pandas  
NumPy  
Scikit learn  
XGBoost  
Matplotlib  
Seaborn  

How to Run

Install required libraries using requirements file  
Run the notebook in notebooks folder  
Use main.py for running the final model  

Conclusion

The project demonstrates that combining machine learning with threshold optimization improves credit risk prediction and makes the model suitable for real world financial applications
