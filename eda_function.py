import pandas as pd

# Load the dataset
df = pd.read_csv('GlobalLandTemperaturesByCity.csv')

# View dataset overview
print(df.head())
print(df.info())

# Summary statistics
print(df.describe())  # Numerical statistics
print(df.describe(include='object'))

#visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of average temperatures
plt.figure(figsize=(8, 5))
sns.histplot(df['AverageTemperature'], bins=50, kde=True)
plt.title('Distribution of Average Temperatures')
plt.xlabel('Average Temperature')
plt.ylabel('Frequency')
plt.show()

# Box plot for AverageTemperature by Country
plt.figure(figsize=(15, 6))
sns.boxplot(x='Country', y='AverageTemperature', data=df)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title('Average Temperature by Country')
plt.show()


# Correlation heatmap
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation
correlation = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Count missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# Box plot for Average Temperature
plt.figure(figsize=(8, 5))
sns.boxplot(df['AverageTemperature'])
plt.title('Outliers in Average Temperature')
plt.show()

# Use IQR method for outlier detection
Q1 = df['AverageTemperature'].quantile(0.25)
Q3 = df['AverageTemperature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['AverageTemperature'] < (Q1 - 1.5 * IQR)) | 
              (df['AverageTemperature'] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers: {len(outliers)}")



# Distribution of numerical features
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.show()

# Unique value counts for categorical features
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col} - Unique values: {df[col].nunique()}")


# Data types and unique values
print(df.dtypes)
print(df.nunique())


# Convert Date column to datetime
df['dt'] = pd.to_datetime(df['dt'])

# Group by year and calculate mean temperature
df['Year'] = df['dt'].dt.year
yearly_trend = df.groupby('Year')['AverageTemperature'].mean()

# Plot temperature trend over years
plt.figure(figsize=(10, 6))
yearly_trend.plot()
plt.title('Yearly Trend of Average Temperature')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.show()



# Aggregation by Region
# Check column names
print("Columns in dataset:", df.columns)

# Ensure 'Region' column exists
if 'Region' not in df.columns:
    print("'Region' column not found. Creating it from 'Country'.")
    region_mapping = {
        'United States': 'North America',
        'Canada': 'North America',
        'India': 'Asia',
        'France': 'Europe',
        # Add more mappings as needed
    }
    df['Region'] = df['Country'].map(region_mapping)

# Perform groupby operation
region_agg = df.groupby('Region')['AverageTemperature'].mean()
print(region_agg)


# Aggregation by Country
country_agg = df.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False)
print(country_agg.head())

# Bar plot of top 10 countries
top_countries = country_agg.head(10)
top_countries.plot(kind='bar', figsize=(10, 5))
plt.title('Top 10 Countries by Average Temperature')
plt.ylabel('Average Temperature')
plt.show()



# Pairplot for selected features
sns.pairplot(df, vars=['AverageTemperature', 'AverageTemperatureUncertainty'])
plt.suptitle('Pairwise Analysis', y=1.02)
plt.show()



# Temperature trend in a specific country
country_data = df[df['Country'] == 'United States']
country_trend = country_data.groupby('Year')['AverageTemperature'].mean()

plt.figure(figsize=(10, 5))
country_trend.plot()
plt.title('Yearly Temperature Trend in the United States')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.show()


#_______________________________________________________________________________________
# Fill missing values in numerical columns
df['AverageTemperature'] = df['AverageTemperature'].fillna(df['AverageTemperature'].mean())
df['AverageTemperatureUncertainty'] = df['AverageTemperatureUncertainty'].fillna(df['AverageTemperatureUncertainty'].mean())

from sklearn.preprocessing import LabelEncoder

# Label Encoding for City and Country
label_encoder_city = LabelEncoder()
label_encoder_country = LabelEncoder()

df['City_Encoded'] = label_encoder_city.fit_transform(df['City'])
df['Country_Encoded'] = label_encoder_country.fit_transform(df['Country'])



# from sklearn.preprocessing import MinMaxScaler
def parse_coordinate(coord):
    """Convert a coordinate with directional indicator to a float."""
    if 'N' in coord or 'E' in coord:
        return float(coord[:-1])  # Remove the last character ('N' or 'E') and convert to float
    elif 'S' in coord or 'W' in coord:
        return -float(coord[:-1])  # Remove the last character and make it negative
    else:
        raise ValueError(f"Invalid coordinate format: {coord}")

# # Apply the function to the Latitude and Longitude columns
df['Latitude'] = df['Latitude'].apply(parse_coordinate)
df['Longitude'] = df['Longitude'].apply(parse_coordinate)


# from sklearn.preprocessing import MinMaxScaler

# # Initialize the scaler
# scaler = MinMaxScaler()

# # Scale the numerical columns
# df[['Latitude_Scaled', 'Longitude_Scaled']] = scaler.fit_transform(df[['Latitude', 'Longitude']])
import pandas as pd

# Assuming df is your DataFrame and you have identified the column with issues
# Clean the column by removing the non-numeric characters
# df['your_column'] = df['your_column'].replace(r'[^\d.]+', '', regex=True)

# Convert the column to numeric (if necessary)
# df['your_column'] = pd.to_numeric(df['your_column'], errors='coerce')  # 'coerce' will replace invalid parsing with NaN
# df['AverageTemperature'] = df['AverageTemperature'].replace(r'[^\d.]+', '', regex=True)
# Save the preprocessed data to a new CSV file
# Clean Latitude and Longitude columns by removing non-numeric characters
df['Latitude'] = df['Latitude'].replace(r'[^\d.-]+', '', regex=True)
df['Longitude'] = df['Longitude'].replace(r'[^\d.-]+', '', regex=True)

# # Convert Latitude and Longitude to numeric (float)
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')


df['AverageTemperature'] = df['AverageTemperature'].fillna(df['AverageTemperature'].mean())
df['AverageTemperatureUncertainty'] = df['AverageTemperatureUncertainty'].fillna(df['AverageTemperatureUncertainty'].mean())


# View dataset overview
print(df.head())
print(df.info())

# Summary statistics
print(df.describe())  # Numerical statistics
print(df.describe(include='object')) 
# print(df.columns)
df.to_csv('preprocessed_climate_data.csv', index=False)