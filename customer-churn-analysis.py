# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options to see more columns
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('path/to/your/telco_customer_churn.csv')

# Get a quick overview of the dataset
print("Dataset shape:", df.shape)  # Shows number of rows and columns

# Display the first few rows to see what the data looks like
print("\nFirst 5 rows:")
print(df.head())

# Check data types and missing values
print("\nDataset info:")
print(df.info())

# Get statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values by column:")
print(df.isnull().sum())

# For categorical columns, check unique values and counts
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True) * 100)  # Shows percentages

# Basic visualization of churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Explore numeric variables relationship with churn
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f'{col} vs Churn')
    plt.show()

# Explore categorical variables relationship with churn
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')  # Remove target variable if it's categorical
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, hue='Churn', data=df)
    plt.title(f'{col} vs Churn')
    plt.xticks(rotation=45)
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

# Select specific features that are important for the model
selected_features = ['Contract', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                     'InternetService', 'OnlineSecurity', 'TechSupport', 
                     'PaymentMethod', 'PaperlessBilling']

# Create feature dataset with only selected columns
X = df[selected_features]

# Target variable remains the same
y = df['Churn']

# Handle categorical variables in the selected features
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance (for logistic regression)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Importance')
plt.show()

# Try a Random Forest model for better performance
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Make the model actionable
# Identify customers with high churn probability
probabilities = rf_model.predict_proba(X_test)

# Create DataFrame with customer IDs and their churn probabilities
customer_risk = pd.DataFrame({
    'Customer_ID': df.loc[y_test.index, 'CustomerID'] if 'CustomerID' in df.columns else y_test.index,
    'Churn_Probability': probabilities[:, 1]
})

# Sort by probability to identify highest-risk customers
high_risk_customers = customer_risk.sort_values(by='Churn_Probability', ascending=False).head(100)
print("Top 10 High-Risk Customers:")
print(high_risk_customers.head(10))

# Feature importance for Random Forest
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
rf_importance = rf_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importance)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Analyze how contract type affects churn prediction
if 'Contract_One year' in X.columns and 'Contract_Two year' in X.columns:
    contract_features = [col for col in X.columns if 'Contract_' in col]
    print("\nContract type coefficients (Logistic Regression):")
    contract_coef = coefficients[coefficients['Feature'].isin(contract_features)]
    print(contract_coef)
    
    print("\nContract type importance (Random Forest):")
    contract_imp = rf_importance[rf_importance['Feature'].isin(contract_features)]
    print(contract_imp)
