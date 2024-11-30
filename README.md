# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data = pd.read_csv(url, parse_dates=["Date"], index_col="Date")


# Display the first few rows of the dataset
print("Dataset preview:")
print(data.head())


# Basic dataset exploration
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())



# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(data['Temp'], label='Daily Minimum Temperature', color='blue')
plt.title('Daily Minimum Temperatures (1981-1990)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# Adding rolling averages for trend analysis
data['7-day Moving Avg'] = data['Temp'].rolling(window=7).mean()
data['30-day Moving Avg'] = data['Temp'].rolling(window=30).mean()



# Plot rolling averages
plt.figure(figsize=(12, 6))
plt.plot(data['Temp'], label='Original Data', color='blue', alpha=0.5)
plt.plot(data['7-day Moving Avg'], label='7-day Moving Average', color='orange')
plt.plot(data['30-day Moving Avg'], label='30-day Moving Average', color='red')
plt.title('Temperature Trends with Moving Averages', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# Calculate yearly averages for trend analysis
data['Year'] = data.index.year
yearly_avg = data.groupby('Year')['Temp'].mean()



# Long-term trend: Moving Average (365-day window)
data['365-day Moving Avg'] = data['Temp'].rolling(window=365).mean()



# Plot yearly average temperatures (Yearly Trends)
plt.figure(figsize=(12, 6))
plt.plot(yearly_avg, marker='o', label='Yearly Average Temperature', color='green')
plt.title('Yearly Average Temperatures (1981-1990)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()



# Plot overall trend with a 365-day moving average
plt.figure(figsize=(12, 6))
plt.plot(data['Temp'], label='Daily Minimum Temperature', color='blue', alpha=0.5)
plt.plot(data['365-day Moving Avg'], label='365-day Moving Average', color='red', linewidth=2)
plt.title('Overall Temperature Trend (1981-1990)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# Seasonal Decomposition

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(data['Temp'], model='additive', period=365)


# Observed data
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed', color='blue')
plt.legend(loc='upper right')
plt.title('Observed Data')



# Trend
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend(loc='upper right')
plt.title('Trend')


# Seasonal component
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper right')
plt.title('Seasonality')


# Residuals
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='purple')
plt.legend(loc='upper right')
plt.title('Residuals')


# Seasonal boxplot
data['Month'] = data.index.month
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Month'], y=data['Temp'], palette="coolwarm")
plt.title('Seasonal Temperature Trends (Monthly)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.grid(alpha=0.3)
plt.show()


plt.tight_layout()
plt.show()
