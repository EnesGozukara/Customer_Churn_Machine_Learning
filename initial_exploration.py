
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


pd.set_option('display.max_columns', None)

df = pd.read_csv("C:\\Users\\enesg\\OneDrive\\Masaüstü\\Data Analysis\\Customer_Churn_Rates\\Telco_Customer_Churn_Dataset.csv")

print("Dataset Shape:", df.shape)

print(df.head())

print(df.info())

print(df.describe())

# As value counts with only unique ones , Churn Rate displayed with binary coding 0-1 (Yes - No)

#Getting Unique Rates for Churn Generally

print(df['Churn'].value_counts(normalize = True) * 100)

# Scores : No - 73.46 , Yes - 26.53 / Refers To %26.53 Churn Rate depend on statistic.

plt.figure(figsize = (8,6))
sns.countplot(x = 'Churn', data = df)
plt.title('Churn Distribution')
plt.show()

# Analyze relationship between Contract and Churn
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print(contract_churn)

# Visualize it
plt.figure(figsize=(10, 6))
contract_churn['Yes'].sort_values().plot(kind='bar', color='coral')
plt.title('Churn Percentage by Contract Type')
plt.ylabel('Churn Percentage')
plt.xlabel('Contract Type')
plt.show()