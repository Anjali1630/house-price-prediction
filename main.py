import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🏠 House Price Prediction Project")
print("=" * 40)

print("\n📊 Loading California Housing Dataset...")
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")

print("\n🔍 Data Overview:")
print(df.head())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n📊 Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(df['PRICE'], bins=30, color='skyblue', alpha=0.7)
axes[0, 0].set_title('House Price Distribution')
axes[0, 0].set_xlabel('Price ($1000s)')
axes[0, 0].set_ylabel('Frequency')

correlation = df.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Matrix')

axes[1, 0].scatter(df['HouseAge'], df['PRICE'], alpha=0.6, color='green')
axes[1, 0].set_xlabel('House Age (years)')
axes[1, 0].set_ylabel('Price ($100k)')
axes[1, 0].set_title('Price vs House Age')

axes[1, 1].scatter(df['MedInc'], df['PRICE'], alpha=0.6, color='red')
axes[1, 1].set_xlabel('Median Income ($10k)')
axes[1, 1].set_ylabel('Price ($100k)')
axes[1, 1].set_title('Price vs Median Income')

plt.tight_layout()
plt.show()

print("\n🤖 Preparing Data for Machine Learning...")

X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("\n🎯 Training Linear Regression Model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

print("\n🔮 Making Predictions...")

y_pred = model.predict(X_test)

print("\n📊 Model Performance:")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Model Accuracy: {r2*100:.1f}%")

print("\n📈 Visualizing Results...")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

print("\n⭐ Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_)
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

print("\n🏡 Sample Prediction:")
print("Using the first house from test set...")

sample_house = X_test.iloc[0:1]
predicted_price = model.predict(sample_house)[0]
actual_price = y_test.iloc[0]

print(f"Predicted Price: ${predicted_price:.2f} (hundreds of thousands)")
print(f"Actual Price: ${actual_price:.2f} (hundreds of thousands)")
print(f"Difference: ${abs(predicted_price - actual_price):.2f}")
