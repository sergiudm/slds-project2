import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
file_path = "assets/life_indicator_2008-2018.xlsx"
excel_data = pd.ExcelFile(file_path)

# Prepare data for a single country
country = "Albania"
years = [int(y) for y in excel_data.sheet_names]
life_data = {}

for year in years:
    df = excel_data.parse(str(year))
    row = df[df["Country Name"] == country]
    if not row.empty:
        life_data[year] = row["Life expectancy at birth, total (years)"].values[0]

# Convert to arrays
X = np.array(list(life_data.keys())).reshape(-1, 1)
y = np.array(list(life_data.values())).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for test set and 2025
y_pred_test = model.predict(X_test)
y_2025 = model.predict(np.array([[2025]]))[0][0]

# Evaluation
rmse = root_mean_squared_error(y_test, y_pred_test)
print(f"Predicted Life Expectancy in 2025 for {country}: {y_2025:.2f} years")
print(f"Model RMSE: {rmse:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="green", label="Trend Line")
plt.scatter([2025], [y_2025], color="red", label=f"2025 Prediction: {y_2025:.2f}")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title(f"Life Expectancy Prediction for {country}")
plt.legend()
plt.grid(True)
plt.savefig(f"results/task1bonus/life_expectancy_prediction_{country}.png")
