
# ML - Linear Regression - Assignment 1

## Submitted by
**Pradyumn Sharma**  
**Roll No:** 23MT0263
**GMAIL:** 23mt0263@iitism.ac.in

## Introduction

This assignment focuses on implementing linear regression techniques using Python. The primary objective is to build and evaluate a linear regression model to understand and predict the relationship between independent variables and the dependent variable.

## Dataset

The dataset used for this assignment is `database.csv`, which contains several features (`x1` to `x8`) and one target variable (`y`). The dataset is loaded and explored to identify relationships, perform data preprocessing, and apply linear regression techniques.

## Data Collection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Pradyumn Sharma\Desktop\MACHINE LEARNING ASSIGNMENT\ASSIGNMENT 1\database.csv")
```

## Data Exploration

```python
# Display dataset structure
df.columns
df.shape
df.describe()
df.info()
df.memory_usage()
df.isnull().sum()

# Renaming columns for clarity
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
df.head(5)
```

### Visualization and Correlation Analysis

```python
# KDE plots for each feature
for i in range(0, 8):
    plt.figure(figsize=(1, 1))
    sns.kdeplot(df.iloc[:, i])
    plt.show()

# Scatter plots of features vs target variable
for i in df.columns:
    plt.figure(figsize=())
    plt.scatter(df[i], df["y"], label=i)
    plt.show()

# Correlation coefficients
for i in df.columns:
    print(i, pd.DataFrame(np.corrcoef(df[i], df['y']))[0][1])

# Correlation among features
df.iloc[0:, 0:8].corr()
sns.heatmap(df.iloc[:, 0:8].corr(), annot=True)
plt.show()
sns.pairplot(df)
```

### Data Standardization

```python
dfstd = df.iloc[:, 0:8]
dfstd = (dfstd - dfstd.mean()) / dfstd.std()

plt.hist(df['y'], bins=50)
plt.show()
```

### Outlier Detection

```python
plt.figure(figsize=(20, 5))
plt.boxplot(dfstd, labels=dfstd.columns)
plt.show()

# Z-Test for outliers in 'x8'
print(df['x8'].unique())
a = [i for i in df['x8'] if (int(i) - df['x8'].mean()) / df['x8'].std() > 3]
pd.Series(a).unique()
```

## Data Preparation

```python
# Define independent and dependent variables
x = dfstd
y = df['y']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

## Model Training and Evaluation

### Using Least Squares Method

```python
x_train_u = x_train.copy()
x_train_u.insert(0, 'Intercept', 1)
x_t = x_train_u.T
x_ty = x_t @ y_train
Co = pd.DataFrame(np.linalg.inv(x_t @ x_train_u))
Coeff = pd.Series(np.dot(Co, x_ty))
print(Coeff)
```

### Using Scikit-Learn

```python
from sklearn.linear_model import SGDRegressor as sr
regr = sr(max_iter=10000)
regr.fit(x_train, y_train)
print(pd.Series(regr.coef_))

from sklearn.linear_model import LinearRegression
Reg = LinearRegression()
Reg.fit(x_train, y_train)
print(pd.Series(Reg.coef_))
print(Reg.intercept_)
```

### Model Testing

```python
x_test.insert(0, 'Intercept', 1)
y_pred = np.dot(x_test, Coeff)
pd.DataFrame({'y_pred': y_pred, 'y_act': y_test.reset_index(drop=True)})
```

## Assumption Validation

### Linearity (F-Test)

```python
import scipy.stats
print(scipy.stats.f.ppf(q=0.95, dfn=len(df.columns)-1, dfd=df.shape[0]-(len(df.columns)-1)-1))
```

### Test for Individual Regression Coefficient (t-Test)

```python
print(scipy.stats.t.ppf(q=1-.05/2, df=df.shape[0]-len(df.columns)-2))
```

### Model Retraining After Removing Features

```python
x_ = x_train[['x1', 'x2', 'x3', 'x4', 'x5', 'x8']]
x_ = (x_ - x_.mean()) / x_.std()
x_.insert(0, 'Intercept', 1)
model_ = sm.OLS(y_train, x_).fit()
print(model_.summary())
```

### Model Testing Predicted Values

```python
x__ = x_test[['x1', 'x2', 'x3', 'x4', 'x5', 'x8']]
x__.insert(0, 'Intercept', 1)
ypred = model_.predict(x__)
print(ypred)
```

### Residual Analysis

```python
Residue = y_test - ypred
plt.figure(figsize=(3, 3))
plt.scatter(Residue, ypred)
plt.show()
```

### Model Adequacy

```python
import scipy.stats
print(scipy.stats.f.ppf(q=0.95, dfn=len(df.columns)-1, dfd=df.shape[0]-(len(df.columns)-1)-1))
```

### Normality of Y

```python
plt.figure(figsize=(5, 5))
sns.kdeplot(y)
plt.show()
```

### Homoscedasticity

```python
plt.figure(figsize=(3, 3))
plt.scatter(Residue, ypred)
```

### Multi-Collinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
for i in range(len(x__.columns)):
    vif_ = vif(x__, i)
    print(x__.columns[i], vif_)
```

### Auto-Correlation

```python
from statsmodels.stats.stattools import durbin_watson as d_w
dw = d_w(Residue)
print(dw)
```

### Linear Regression Using Gradient Descent

```python
from sklearn.linear_model import SGDRegressor as sr
regr = sr()
regr.fit(x_train, y_train)
print(pd.Series(regr.predict(x_test), y_test))
```

## Conclusion

- The linear regression model was successfully implemented and evaluated using multiple techniques.
- Feature removal improved model performance, as evidenced by increased R-squared values.
- The assumptions of linear regression were validated, including linearity, normality, homoscedasticity, and no multicollinearity.
