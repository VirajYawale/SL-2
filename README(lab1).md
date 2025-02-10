

### Lab-1:  Linear Regression and Polynomial Regression with Python using scikit-learn

Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values based on input features. It establishes a linear relationship between independent variables (X) and dependent variable (Y).
(refer: https://youtu.be/8jazNUpO3lQ?si=0C1jV95027nM9wo7)

Ex: Predicting House Prices 🏡 :  Given the size of a house, predict its price.

***
#### In Lab-1 file:
**Simple Linear Regression:** 

- Uses a small housing dataset (area vs. price).
- Implements Linear Regression using ```sklearn.linear_model.LinearRegression```
- Splits data into training and testing sets.
- Trains the model and predicts house prices.
- Evaluates performance using *Mean Squared Error (MSE) and R² score*.
- Visualizes predictions with scatter plots and regression lines.

**Scikit-learn (or sklearn) is a powerful Python library used for machine learning and data science. It provides efficient tools for data preprocessing, model training, evaluation, and predictions.**

**Note:**

```bash
# Simulated dataset: Housing area vs. price
data = {
    'Area': [500, 750, 1000, 1250, 1500],
    'Price': [100, 150, 200, 250, 300]
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['Area']]  # Independent variable (area)
y = df['Price']   # Dependent variable (price)
```

**1.Splitting the data into test and train:**
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```
- Splits X and y into training (60%) and testing (40%) data.
- random_state=42 ensures reproducibility (same split every time the code runs).
If random_state is not set, the dataset will be split randomly every time you run the code.

**2. Train the model:**
```bash
model = LinearRegression()
model.fit(X_train, y_train)
```
- We can use other regression model to train the model (ex: **Polynomial Regression, Decision Trees**)
- **according the use of model we will train the model** (ex: to classify categories (e.g., spam detection, disease prediction), use these: **Logistic Regression (For Binary Classification) **Yes or No** )**
- **Classification is a supervised learning technique where a model learns to assign labels (categories) to input data based on features. ***The goal is to predict which category a new data point belongs to.*****
- ```model.fit(X_train, y_train)``` :Finds the best-fit line by calculating the slope (m) and intercept (b) using Least Squares Method.

**3. Making Predictions:**
```bash
y_pred = model.predict(X_test)  # Predict prices for test set
```
- Uses the trained model to predict Price (y_pred) for new Area values (X_test).

**7. Displaying Model Parameters:**
```bash
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

```
- ```model.coef_```: Returns the slope (m) of the best-fit line. There is only one feature here (i.e area) so, we pass [0] in it.
- ```model.intercept_```: Returns the y-intercept (b) of the best-fit line.

**8. Displaying Predictions vs. Actual Values:**
```bash
print(f"Test set predictions: {y_pred}")
print(f"Actual test values: {y_test.values}")
```

**9. Plotting the Regression Line:**
```bash
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X, model.predict(X), color='red', label='Predicted line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (thousands)')
plt.legend() #it is use to lable the notations.
plt.title('Linear Regression on Housing Data')
plt.show()
```

**10. Evaluating Model Performance:**
```bash
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

```
-  ```mean_squared_error(y_test, y_pred):``` Measures average squared difference between actual and predicted values.
- Lower MSE means better accuracy.
- ```r2_score(y_test, y_pred):``` Measures how well the model explains the variance in data.
- R² closer to 1 means a better fit.

**MSE (Mean Squared Error) → Measures how far the predictions are from actual values.** (Lower MSE means better accuracy.)

**R² Score (R-Squared or Coefficient of Determination) → Measures how well the model explains the data.** (R² closer to 1 means a better fit.)

***
***
***
***


**In lab-1 file:**
we perform performs Linear Regression on the Boston Housing Dataset to predict house prices (MEDV) based on multiple features.

[Explaination link!](https://chatgpt.com/share/67a8edf1-f818-8006-bb9a-4310a9cc2ffe)

Also, the Polynomial Regression [Explaination link!](https://chatgpt.com/share/67a8ed76-9f00-8006-8e82-e0b1851f2b87)

Use of pipeline in code [Explanation link!](https://chatgpt.com/share/67a8ea52-3298-8006-ad59-9e594f3c7be0)


***
***
***
***

**In lab-1 Assignment:**
We used Ridge and Lasso Regression

Both Ridge and Lasso are types of regularized regression techniques that improve linear regression by reducing overfitting and enhancing generalization.

***
***
***
***

**Classification :**
Classification is a supervised learning technique where a model learns to assign labels (categories) to input data based on features.

**The goal is to predict which category a new data point belongs to.**
- Binary Classification - **Logistic Regression**

- Multiclass Classification (More than Two Classes: Each sample belongs to only one category out of multiple classes.) - **Softmax Regression (Multinomial Logistic Regression), k-Nearest Neighbors (KNN)** - **Use:** Handwritten Digit Recognition (Digits 0-9)

- Multilabel Classification (Multiple Labels per Sample): **Neural Networks** - **Use:** Movie Genre Prediction (A movie can be Action, Comedy, and Sci-Fi at the same time).

- Imbalanced Classification (One Class is Much More Common) - **SMOTE (Synthetic Minority Oversampling Technique)** - **Use:** Fraud Detection (99% genuine transactions, 1% fraud).
