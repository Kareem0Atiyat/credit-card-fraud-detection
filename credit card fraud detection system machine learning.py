#!/usr/bin/env python
# coding: utf-8

# # credit card fraud detection system machine learning

# In[ ]:


pip install pandas numpy matplotlib seaborn scikit-learn


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('creditcard.csv')

# Normalize 'Amount' and drop unnecessary columns
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ##  Conclusion
# 
# This credit card fraud detection project demonstrates how machine learning can be applied to solve real-world classification problems, especially with highly imbalanced data.
# 
# Key takeaways:
# 1. **Data imbalance is a critical issue** — with fraud cases being less than 0.2% of the dataset, metrics like precision, recall, and F1-score are more important than accuracy.
# 2. **Random Forest performed well** for this binary classification task due to its robustness and ability to handle imbalance (with class weights or oversampling if needed).
# 3. **Feature scaling** and **data preprocessing** (like dropping 'Time', scaling 'Amount') played a key role in model performance.
# 4. **Confusion Matrix** and **classification report** helped us understand false positives and false negatives — crucial for fraud detection.
# 
# 
# This project strengthened my practical skills in Python, machine learning, and evaluation techniques for high-impact business problems.
# 

# In[ ]:




