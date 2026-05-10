import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/Ames_Housing_Data.tsv", sep="\t")

print("\nFIRST 5 ROWS:")
print(df.head())

# =========================
# BASIC EDA
# =========================

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns)

print("\nDATASET INFO:")
print(df.info())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# =========================
# MISSING VALUES
# =========================

print("\nMISSING VALUES:")
print(df.isnull().sum().sort_values(ascending=False))

# =========================
# HEATMAP
# =========================

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# =========================
# LOG TRANSFORMATION
# =========================

df['SalePrice_Log'] = np.log1p(df['SalePrice'])

plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice_Log'], kde=True)
plt.title("Log Transformed SalePrice")
plt.show()

# =========================
# STANDARD SCALING
# =========================

scaler = StandardScaler()

df['Lot Area_scaled'] = scaler.fit_transform(df[['Lot Area']])

print("\nSCALED FEATURES:")
print(df[['Lot Area', 'Lot Area_scaled']].head())

# =========================
# MINMAX SCALING
# =========================

minmax = MinMaxScaler()

df['Lot Area_minmax'] = minmax.fit_transform(df[['Lot Area']])

print("\nMINMAX FEATURES:")
print(df[['Lot Area_minmax']].head())

# =========================
# LABEL ENCODING
# =========================

encoder = LabelEncoder()

df['Neighborhood_encoded'] = encoder.fit_transform(df['Neighborhood'])

print("\nENCODED FEATURES:")
print(df[['Neighborhood', 'Neighborhood_encoded']].head())

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(14,10))

sns.heatmap(df.corr(numeric_only=True),
            cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

# =========================
# OUTLIER DETECTION
# =========================

plt.figure(figsize=(8,5))

sns.boxplot(x=df['SalePrice'])

plt.title("SalePrice Outliers")

plt.show()

# =========================
# SAVE PROCESSED DATA
# =========================

df.to_csv("processed_data.csv", index=False)

print("\nProcessed dataset saved successfully!")