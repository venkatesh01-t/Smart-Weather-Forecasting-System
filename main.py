import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
df = pd.read_csv('dataset/seattle-weather.csv')  # Replace 'your_dataset.csv' with the actual file path

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Prepare data for modeling
y = df['weather']  # Target variable is changed to 'weather'
X = df[['date', 'precipitation', 'temp_max', 'temp_min', 'wind']]

# Add additional time-related features
X['dayofweek'] = df['date'].dt.dayofweek
X['day'] = df['date'].dt.day
X['month'] = df['date'].dt.month
X['year'] = df['date'].dt.year

# Time Series Split
tscv = TimeSeriesSplit(5)
acc_score = []

# Random Forest Model
for i, (train_ind, test_ind) in enumerate(tscv.split(X)):
    X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
    X_test, y_test = X.iloc[test_ind], y.iloc[test_ind]

    model = RandomForestClassifier()
    model.fit(X_train.drop('date', axis=1), y_train)

    y_pred = model.predict(X_test.drop('date', axis=1))
    acc = accuracy_score(y_test, y_pred)
    acc_score.append(acc)

    print(f'{i}th iter:')
    print(classification_report(y_test, y_pred))

# Save the model using pickle
with open('model/weather_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Generating hypothetical future data for prediction
future_dates = pd.date_range(start='2024-03-01', end='2024-03-10', freq='D')
future_precipitation = np.random.uniform(0, 10, size=len(future_dates))
future_temp_max = np.random.uniform(20, 30, size=len(future_dates))
future_temp_min = np.random.uniform(10, 20, size=len(future_dates))
future_wind = np.random.uniform(5, 15, size=len(future_dates))

future_data = pd.DataFrame({
    'date': future_dates,
    'precipitation': future_precipitation,
    'temp_max': future_temp_max,
    'temp_min': future_temp_min,
    'wind': future_wind
})

# Add time-related features
future_data['dayofweek'] = future_data['date'].dt.dayofweek
future_data['day'] = future_data['date'].dt.day
future_data['month'] = future_data['date'].dt.month
future_data['year'] = future_data['date'].dt.year

# Convert 'date' column to datetime
future_data['date'] = pd.to_datetime(future_data['date'])

# Make predictions
future_data['weather_prediction'] = model.predict(future_data.drop('date', axis=1))

# Data Visualization Graphs

# 1. Line plot for temperature over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='temp_max', data=df, label='Max Temperature')
sns.lineplot(x='date', y='temp_min', data=df, label='Min Temperature')
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('model/1.png')
plt.show()

# 2. Bar plot for precipitation
plt.figure(figsize=(8, 6))
df['month'] = df['date'].dt.month  # Add this line to explicitly create a 'month' column
sns.barplot(x='month', y='precipitation', data=df)
plt.title('Monthly Precipitation')
plt.xlabel('Month')
plt.ylabel('Precipitation')
plt.savefig('model/2.png')
plt.show()

# 3. Count plot for weather types
plt.figure(figsize=(8, 6))
sns.countplot(x='weather', data=df)
plt.title('Weather Types Distribution')
plt.xlabel('Weather Type')
plt.ylabel('Count')
plt.savefig('model/3.png')
plt.show()

# 4. Pie chart for the distribution of predicted weather in the future
plt.figure(figsize=(8, 8))
future_data['weather_prediction'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Future Weather Predictions Distribution')
plt.savefig('model/4.png')
plt.show()

# 5. Box plot for temperature distribution by month
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='temp_max', data=df)
plt.title('Temperature Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Max Temperature')
plt.savefig('model/5.png')
plt.show()

# 6. Pair plot for relationships between numerical features
sns.pairplot(df[['temp_max', 'temp_min', 'precipitation', 'wind']])
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.savefig('model/6.png')
plt.show()

# 7. Heatmap for correlation between numerical features
plt.figure(figsize=(10, 8))
correlation_matrix = df[['temp_max', 'temp_min', 'precipitation', 'wind']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('model/7.png')
plt.show()

# 8. Line plot for future precipitation
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='precipitation', data=future_data)
plt.title('Future Precipitation Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation')
plt.savefig('model/8.png')
plt.show()

# 9. Scatter plot for temperature vs. wind speed
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp_max', y='wind', data=df, hue='weather')
plt.title('Temperature vs. Wind Speed')
plt.xlabel('Max Temperature')
plt.ylabel('Wind Speed')
plt.savefig('model/9.png')
plt.show()

# 10. Violin plot for temperature distribution by weather type
plt.figure(figsize=(10, 6))
sns.violinplot(x='weather', y='temp_max', data=df)
plt.title('Temperature Distribution by Weather Type')
plt.xlabel('Weather Type')
plt.ylabel('Max Temperature')
plt.savefig('model/10.png')
plt.show()
