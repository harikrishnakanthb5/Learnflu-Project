# ---------------------------------------------------------
# PROJECT: Find Linear Relationship between Media and Sales
# ---------------------------------------------------------

# a. Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# b. Import the data
# ---------------------------------------------------------

# Read the uploaded text dataset
data = pd.read_csv("a_batch.txt", sep=r"\s+")

print("Dataset:")
print(data)

# ---------------------------------------------------------
# c. Analysis of data
# ---------------------------------------------------------

print("\nFirst 5 records:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nStatistical analysis:")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

print("\nCorrelation between Media and Sales:")
print(data["media"].corr(data["sales"]))

# ---------------------------------------------------------
# d. Declare feature variable and target variable
# ---------------------------------------------------------

X = data[["media"]]    # Feature variable
y = data["sales"]      # Target variable

print("\nFeature variable X:")
print(X.head())

print("\nTarget variable y:")
print(y.head())

# ---------------------------------------------------------
# e. Plot scatter plot between X and y
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(X, y)

plt.xlabel("Media")
plt.ylabel("Sales")
plt.title("Scatter Plot between Media and Sales")

plt.show()

# ---------------------------------------------------------
# f. Checking and reshaping of X and y
# ---------------------------------------------------------

print("\nShape of X before reshaping:")
print(X.shape)

print("\nShape of y:")
print(y.shape)

# Convert X and y into NumPy arrays
X = np.array(X)
y = np.array(y)

# Reshape X
X = X.reshape(-1, 1)

# Reshape y
y = y.reshape(-1, 1)

print("\nShape of X after reshaping:")
print(X.shape)

print("\nShape of y after reshaping:")
print(y.shape)

# ---------------------------------------------------------
# Split the data into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ---------------------------------------------------------
# g. Apply Linear Regression Model
# ---------------------------------------------------------

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Display coefficients
print("\nRegression Coefficient:", model.coef_[0][0])
print("Regression Intercept:", model.intercept_[0])

# Regression equation
print("\nRegression Equation:")
print(
    "Sales = {:.3f} + {:.3f} * Media".format(
        model.intercept_[0],
        model.coef_[0][0]
    )
)

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nR-squared:", r2)
print("Mean Squared Error:", mse)

# ---------------------------------------------------------
# Plot the Regression Line
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

# Scatter plot of original data
plt.scatter(X, y, label="Actual Data")

# Regression line
plt.plot(
    X,
    model.predict(X),
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Media")
plt.ylabel("Sales")
plt.title("Linear Regression: Media vs Sales")
plt.legend()
plt.grid(True)

plt.show()