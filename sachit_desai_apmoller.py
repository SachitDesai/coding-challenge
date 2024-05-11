# -*- coding: utf-8 -*-
"""sachit_desai_apmoller.ipynb


# Sachit Desai coding challenge notebook

## 1. import libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

"""## 2. load datasets"""

#loading the csv's into dataframes
training_df = pd.read_csv("https://raw.githubusercontent.com/SachitDesai/coding-challenge/main/training-dataset.csv")
testing_df = pd.read_csv("https://raw.githubusercontent.com/SachitDesai/coding-challenge/main/testing-dataset.csv")

"""## 3. understanding the dataset / EDA




"""

training_df.head()

testing_df.head()

print(training_df.describe())
print(training_df.info())

# count of categorical variables
print(training_df['ProductType'].value_counts())
print(training_df['Manufacturer'].value_counts())
print(training_df['Area Code'].value_counts())
print(training_df['Sourcing Channel'].value_counts())
print(training_df['Product Size'].value_counts())
print(training_df['Month of Sourcing'].value_counts())

# checking for null values
training_df.isnull().sum()

# checking all the unique values in each feature
for col in training_df.columns:
    if col != "Sourcing Cost":
        unique_entries = training_df[col].unique()
        print(f"Unique entries for column '{col}':")
        print(unique_entries)
        print()

# checking number of duplicate rows
print("\nNumber of duplicate rows:", training_df.duplicated().sum())

# finding the range of Sourcing Cost
min_sourcing_cost = training_df['Sourcing Cost'].min()
max_sourcing_cost = training_df['Sourcing Cost'].max()
# noticed negative value in sourcing cost
print("Range of Sourcing Cost:")
print("Minimum Sourcing Cost:", min_sourcing_cost)
print("Maximum Sourcing Cost:", max_sourcing_cost)

# printing all negative values in Sourcing Cost
negative_sourcing_cost = training_df[training_df['Sourcing Cost'] < 0]['Sourcing Cost']
print("Negative Sourcing Cost:")
print(negative_sourcing_cost)
negative_sourcing_cost_count=negative_sourcing_cost.count()
print("number of rows with negative sourcing costs: ", negative_sourcing_cost_count)

# List of categorical features
categorical_features = ['ProductType', 'Manufacturer', 'Area Code', 'Sourcing Channel', 'Product Size', 'Product Type', 'Month of Sourcing']

# Create subplots to visualize multiple categorical features
fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(10, 6*len(categorical_features)))

# Loop through each categorical feature and plot its distribution
for i, feature in enumerate(categorical_features):
    sns.countplot(x=feature, data=training_df, ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""## 4. preprocessing and more eda

"""

# finding outliers using box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sourcing Cost', data=training_df)
plt.title("Box Plot of Sourcing Cost")
plt.xlabel("Sourcing Cost")
plt.show()

# z-scores
z_scores = (training_df['Sourcing Cost'] - training_df['Sourcing Cost'].mean()) / training_df['Sourcing Cost'].std()
outliers_zscore = training_df[np.abs(z_scores) > 3]  # threshold of 3 for z-score
print("Outliers identified using z-scores:")
print(outliers_zscore)

# removing outliers using IQR
q1 = training_df['Sourcing Cost'].quantile(0.25)
q3 = training_df['Sourcing Cost'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

cleaned_training_df = training_df[(training_df['Sourcing Cost'] >= lower_bound) & (training_df['Sourcing Cost'] <= upper_bound)]

num_outliers_removed = len(training_df) - len(cleaned_training_df)
print(f"Number of outliers removed: {num_outliers_removed}")

cleaned_training_df.reset_index(drop=True, inplace=True)

# boxplot after removing outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sourcing Cost', data=cleaned_training_df)
plt.title("Box Plot of Sourcing Cost")
plt.xlabel("Sourcing Cost")
plt.show()

cleaned_training_df.head()

# converting Month of Sourcing to datetime in both training and testing datasets
cleaned_training_df['Month of Sourcing'] = pd.to_datetime(cleaned_training_df['Month of Sourcing'], format='%b-%y')
cleaned_training_df['Month of Sourcing'] = cleaned_training_df['Month of Sourcing'].dt.to_period('M').dt.to_timestamp()

testing_df['Month of Sourcing'] = pd.to_datetime(testing_df['Month of Sourcing'], format='%b-%y')
testing_df['Month of Sourcing'] = testing_df['Month of Sourcing'].dt.to_period('M').dt.to_timestamp()

# Set index to Month of Sourcing for both datasets
cleaned_training_df.set_index('Month of Sourcing', inplace=True)
testing_df.set_index('Month of Sourcing', inplace=True)

# plot month of sourcing vs sourcing cost after sorting the months
cleaned_training_df_sorted = cleaned_training_df.sort_values(by='Month of Sourcing')

plt.figure(figsize=(10, 6))
sns.lineplot(x='Month of Sourcing', y='Sourcing Cost', data=cleaned_training_df_sorted)
plt.title('Total Sourcing Cost Over Time')
plt.xlabel('Month of Sourcing')
plt.ylabel('Total Sourcing Cost')
plt.xticks(rotation=45)
plt.show()

# create month and year features for training and testing datasets
cleaned_training_df['Month'] = cleaned_training_df.index.month
cleaned_training_df['Year'] = cleaned_training_df.index.year

testing_df['Month'] = testing_df.index.month
testing_df['Year'] = testing_df.index.year

cleaned_training_df.head()

testing_df.head()

# one hot encoding the training and testing datasets
# have to remove month of sourcing as index first
cleaned_training_df.reset_index(inplace=True)
testing_df.reset_index(inplace=True)

categorical_features = ['ProductType', 'Manufacturer', 'Area Code', 'Sourcing Channel', 'Product Size', 'Product Type']
X_categorical = cleaned_training_df[categorical_features]

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoded_features = encoder.fit_transform(X_categorical)

encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# concatenate encoded features with original dataframe
df_encoded = pd.concat([cleaned_training_df.drop(columns=categorical_features), encoded_df], axis=1)

# this is the final training dataset
df_encoded.head()

X_categorical_test = testing_df[categorical_features]
encoded_features_test = encoder.transform(X_categorical_test)

encoded_df_test = pd.DataFrame(encoded_features_test, columns=encoder.get_feature_names_out(categorical_features))

testing_df_encoded = pd.concat([testing_df.drop(columns=categorical_features), encoded_df_test], axis=1)

# this is the final testing dataset
testing_df_encoded.head()

X_train = df_encoded.drop(columns=['Month of Sourcing', 'Sourcing Cost'])
y_train = df_encoded['Sourcing Cost']

X_test = testing_df_encoded.drop(columns=['Month of Sourcing', 'Sourcing Cost'])
y_test = testing_df_encoded['Sourcing Cost']


"""## 5. model training using LightGBM"""

lgb_regressor = lgb.LGBMRegressor()

lgb_regressor.fit(X_train, y_train)

y_pred_lgb = lgb_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_lgb)
print("Mean Absolute Error (MAE):", mae)
mse = mean_squared_error(y_test, y_pred_lgb)
print("Mean Squared Error (MSE):", mse)
rmse = mean_squared_error(y_test, y_pred_lgb, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)
r2 = r2_score(y_test, y_pred_lgb)
print("R-squared (R2):", r2)
mape = mean_absolute_percentage_error(y_test, y_pred_lgb)
print("Mean Absolute Percentage Error (MAPE):", mape)

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', marker='o')
plt.plot(y_test.index, y_pred_lgb, label='Predicted', marker='x')
plt.xlabel('Time')
plt.ylabel('Sourcing Cost')
plt.title('Actual vs. Predicted Sourcing Cost over Time')
plt.legend()
plt.show()


"""## 6. Explanation

### **a. Cleaning the data**

#### The first thing i noticed when i was exploring the dataset was that there were entries in the training data where the Sourcing Cost had negative values. I then checked for negative values in Sourcing Cost in the Testing data and couldn't find any. Since i have minnimal domain knowledge i was unsure if this was poor data quality or not, so i trained models in both the scenarios. The results i got in both scenarios did not vary in the slightest which is why i decided to keep the negative Sourcing Cost in the training data.

### **b. Outliers**

#### I also trained the models with and without outliers in Sourcing Cost but again the results i got were exactly the same. I decided to remove the outliers.

### **c. Approaches**

####As this is a time series problem, we can solve this problem using Machine Learning Regression to make predictions or using models such as VAR or ARIMA to make forecasts. I have not worked with the latter models, which is why i went ahead with Regression to solve the problem. I used the following models:
1. XGBoost Regressor
2. Random Forest Regressor
3. LightGBM Regressor

I had four main approaches while transforming the dataset:
1. treating the 'Month of Sourcing' as categorical and one-hot encoding it with the rest of the categorical features.
2. treating the 'Month of Sourcing' as categorical and label encoding it with the rest of the categorical features.
3. extracting the month and year from the 'Month of Sourcing' as numerical values, and one-hot encode the categorical features.
4. extracting the month and year from the 'Month of Sourcing' as numerical values, and label encode the categorical features.

Extracting the month and year from the 'Month of Sourcing' as numerical values, and one-hot encoding the categorical features provided the best results overall so that is the pipeline I chose for my final approach.

From the results obtained, it was found that:
- LightGBM had the lowest MAE score of 16.35
- LightGBM had the lowest MSE score of 1002.69
- LightGBM had the lowest RMSE score of 31.67
- Random Forest had the lowest R2 score of 0.61
- Random Forest had the lowest MAPE score of 0.33

Therefore for the final approach, I decided to go with the following pipeline:
1. keep duplicate values
2. remove outliers
3. keep negative values
4. extract numerical 'Month' and 'Year' features from 'Month of Sourcing'
5. encode the categorical features
6. Train and Test XGBoost regressor
7. plot the predictions of Sourcing Cost vs the actual Sourcing Cost for Jun-21
"""