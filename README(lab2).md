
## Lab-2: Regression with Decision Tree
REFER: https://youtu.be/PHxYNGo8NcI?si=-f3gtfmMQK5jzwn1

Here, We use **iris dataset** : The Iris dataset is primarily used for classification tasks in machine learning. It contains data about different species of the iris flower and helps train models to classify them based on their attributes.

There are othe dataset to:

1. **Classification Datasets (Like Iris)**:
- These datasets involve predicting categorical labels (e.g., "Spam" or "Not Spam").
Ex: 
- **Breast Cancer Dataset** (```load_breast_cancer```): Used for breast cancer diagnosis.
- **MNIST Dataset**: Large-scale handwritten digits dataset (extension of load_digits).
- **Titanic Dataset**: Predicts survival on the Titanic.

2. **Regression Datasets (Like California Housing)**
- These datasets involve predicting continuous numerical values.
Ex: 
- **Boston Housing** (```fetch_openml(name="boston")```) (Deprecated): House price prediction dataset.
- **Diabetes Dataset** (```load_diabetes```): Predicts disease progression based on medical data.

3.  **Natural Language Processing (NLP) Datasets**
- These datasets deal with text-based machine learning.
Ex:
- **Spam Detection** (SMS Spam Collection): Classifies SMS messages as spam or not.

4. **Time Series Datasets**
- These datasets involve sequences of data points over time.
Ex:
- **Stock Market Dataset**: Predicts stock prices.
- **COVID-19 Cases Dataset**: Predicts trends in pandemic data

***

**From the Lab-2 File:**

```bash
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
```
**Scikit-learn**:
- ```fetch_california_housing```: Loads California housing dataset (for regression).
- ```load_iris```: Loads Iris dataset (for classification).
- ```train_test_split```: Splits dataset into training/testing sets.
- ```DecisionTreeRegressor```: Decision Tree model for regression tasks.
- ```DecisionTreeClassifier```: Decision Tree model for classification tasks.
- ```plot_tree```: Visualizes decision trees.
- ```mean_squared_error, r2_score```: Evaluate regression models.
- ```accuracy_score, classification_report```: Evaluate classification models.
1. **Loading the California Housing Dataset (Regression):**
```bash
california = fetch_california_housing(as_frame=True)
X_reg = california.data
y_reg = california.target
```
- ```X_reg```: Stores independent features (house attributes like rooms, location, etc.).
- ```y_reg```: Stores the target variable (house price).
- the parameter ```as_frame=True``` is used to return the dataset as a pandas DataFrame instead of NumPy arrays.
2. **Splitting Data for Regression:**
```bash
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.4, random_state=42)
```
- 60% Training data, 40% Testing data (```test_size=0.4```).
- ```random_state=42``` ensures reproducibility.
3. **Initializing and Training the Decision Tree Regressor:**
```bash
regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
regressor.fit(X_train, y_train)
```
- ```DecisionTreeRegressor()```: Creates a Decision Tree model for regression tasks
- ```max_depth=5```: Limits the tree to a maximum depth of 5. (Prevents overfitting by restricting tree growth. Reduces complexity, making the model more generalizable.)
**Example: How the Decision Tree Works**

Let's say we are predicting house prices based on features like median income, population, etc.

Before Training (Raw Data)
| Median Income | Population     | House Price ($)                |
| :-------- | :------- | :------------------------- |
| 5.0 | 1200 | 350,000 |
| 3.5 | 2500 | 180,000 |
| 8.0 | 1000 | 600,000 |

After Training (Decision Tree)

```bash
          Median Income ≤ 4.5?
           /            \
       Yes              No
      /                   \
  Population ≤ 2000?      High price range
   /       \
 Low       Medium  
```
- If Median Income ≤ 4.5, then check Population.
- If Population ≤ 2000, predict low price.
- If Median Income > 4.5, predict high price.

4. **Predit the data and Evaluate the metrics**

```bash

# Predict on test data
y_pred = regressor.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Decision Tree Regressor - MSE: {mse:.2f}, R²: {r2:.2f}") # to gert the 80% more accurecy the mse should be near 0.2..
```

5. **Plot the decision tree:**

```bash
# Plot the decision tree
plt.figure(figsize=(10, 8))
plot_tree(regressor, feature_names=x_reg.columns, filled=True)
plt.title("Decision Tree Regressor")
plt.show()
```

**Similarly for the Classification work of Decision tree:**
```bash
# For Classifier

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Loading california dataset
iris = load_iris(as_frame = True)
x_reg = iris.data
y_reg = iris.target
#for secision tree clasifier
# Split data
X_train, X_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=0.4, random_state=42)
# Initialize and train the classifier
classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
classifier.fit(X_train, y_train)
# Predict on test data
y_pred = classifier.predict(X_test)
# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
# Classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", report)
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree classifier - MSE: {mse:.2f}, R²: {r2:.2f}")
# Plot the decision tree
plt.figure(figsize=(10, 8))
plot_tree(classifier, feature_names=x_reg.columns, filled=True)
plt.title("Decision Tree classifier")
plt.show()
```

***

- **Regression is used when the target variable (output) is continuous.**
- **Classification is used when the target variable (output) is categorical.**

***

- After that we add the noise to it and implement:

```bash
# Adding Gaussian noise to the dataset
#mse might slightly increase and r2 right slightly decrease
noise_factor = 0.5 # Adjust noise level
np.random.seed(42)  # Ensures reproducibility
noise = np.random.normal(loc=0, scale=noise_factor, size=x_reg.shape)
x_reg_noisy = x_reg + noise  # Adding noise to the features
```

Adding Gaussian noise to the dataset is a data augmentation technique used to:

- Improve model robustness – Prevents overfitting by making the model generalize better.
- Simulate real-world imperfections – Real-world data often contains noise due to sensor errors, missing values, or variations.
- Enhance feature variation – Helps prevent the model from memorizing exact patterns and instead forces it to learn general trends.
- If the noisy data was used, it would help the model generalize better but **might also reduce performance slightly.**

***

- **Performance Metrics Explanation**
```bash
# accuracy = No. of correct predictions / Total number of predictions
# precision = true positive / (true positive + false positive)
# recall = true positive / (true positive + false negative) #How many of the actual positive cases were correctly predicted?
# f1 = 2 * (P * R) / (P + R) #The F1-score is a metric used in classification problems to evaluate a model's accuracy.
# macro avg = Average of precision, recall, and F1-score across all classes
# weighted avg = Weighted average based on class distribution

```

Understanding True Positive (TP) & False Positive (FP)
True Positive (TP) and False Positive (FP) are key concepts in classification problems used to evaluate model performance.

**Definitions:**

✅ True Positive (TP)
The model correctly predicts a positive class.

- Example: If we are predicting if a patient has a disease, and the model correctly predicts "Has Disease", this is a TP.

❌ False Positive (FP)
The model incorrectly predicts a positive class.

- Example: If the model predicts "Has Disease" for a patient who is actually healthy, this is a FP. Also called a Type I Error.

❌ False Negative (FN)
The model incorrectly predicts a negative class.

- Example: If the model predicts "No Disease", but the patient actually has the disease, this is a FN. Also called a Type II Error.

✅ True Negative (TN)
The model correctly predicts a negative class.

- Example: If the model correctly predicts "No Disease" for a healthy person, this is a TN.

***

**Lab-2 Assignment:**

This code performs binary classification on the Breast Cancer dataset using two models:

- **Logistic Regression**
- **Decision Tree Classifier**
It also evaluates model performance using various metrics and handles class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique).**
Finally, it plots **ROC curves** to compare model performance.

- **SMOTE (Synthetic Minority Over-sampling Technique) is used to handle imbalanced datasets by generating synthetic data for the minority class.** (Usage: Used before training the model to balance the dataset.)
- **The ROC Curve (Receiver Operating Characteristic Curve) is a graphical representation of a classifier’s performance at different threshold settings.** (Usage: Used after training to analyze model performance.) (Shows trade-off between TPR and FPR.)

