# ============================================
# DATA WAREHOUSE AND ETL PIPELINE
# ============================================

import pandas as pd

print("===================================")
print("STARTING ETL PROCESS")
print("===================================")


# ============================================
# 1. EXTRACT — Load Raw Dataset
# ============================================

df = pd.read_csv("Online Retail.csv", encoding="ISO-8859-1")

print("\nDataset Loaded")
print("Initial Shape:", df.shape)



# ============================================
# 2. TRANSFORM — Data Cleaning
# ============================================

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove duplicates
df = df.drop_duplicates()

# Remove returns and invalid values
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

print("\nAfter Cleaning Shape:", df.shape)



# ============================================
# 3. Convert Data Types
# ============================================

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['CustomerID'] = df['CustomerID'].astype(int)



# ============================================
# 4. Create Revenue Column
# ============================================

df['Revenue'] = df['Quantity'] * df['UnitPrice']



# ============================================
# 5. DATA WAREHOUSE AGGREGATION (WEEKLY FACT TABLE)
# ============================================

weekly_sales = df.resample('W-MON', on='InvoiceDate').agg({

    'Revenue': 'sum',
    'InvoiceNo': 'count',
    'CustomerID': 'nunique',
    'Quantity': 'sum',
    'StockCode': 'nunique'

}).reset_index()


weekly_sales.rename(columns={

    'Revenue': 'Total_Weekly_Sales',
    'InvoiceNo': 'Transaction_Count',
    'CustomerID': 'Customer_Count',
    'Quantity': 'Quantity_Sold',
    'StockCode': 'Unique_Products_Sold'

}, inplace=True)


print("\nWeekly Fact Table Created")



# ============================================
# 6. BUSINESS INTELLIGENCE FEATURES
# ============================================

weekly_sales['Avg_Order_Value'] = (
    weekly_sales['Total_Weekly_Sales']
    / weekly_sales['Transaction_Count']
)

weekly_sales['Avg_Quantity_Per_Transaction'] = (
    weekly_sales['Quantity_Sold']
    / weekly_sales['Transaction_Count']
)

print("BI Metrics Created")



# ============================================
# 7. MACHINE LEARNING FEATURES
# ============================================

weekly_sales = weekly_sales.sort_values('InvoiceDate')


# Lag Features
weekly_sales['Lag_1'] = weekly_sales['Total_Weekly_Sales'].shift(1)
weekly_sales['Lag_2'] = weekly_sales['Total_Weekly_Sales'].shift(2)
weekly_sales['Lag_3'] = weekly_sales['Total_Weekly_Sales'].shift(3)
weekly_sales['Lag_4'] = weekly_sales['Total_Weekly_Sales'].shift(4)
weekly_sales['Lag_5'] = weekly_sales['Total_Weekly_Sales'].shift(5)
weekly_sales['Lag_6'] = weekly_sales['Total_Weekly_Sales'].shift(6)
weekly_sales['Lag_7'] = weekly_sales['Total_Weekly_Sales'].shift(7)
weekly_sales['Lag_8'] = weekly_sales['Total_Weekly_Sales'].shift(8)


# Rolling Features
weekly_sales['Rolling_Mean_4'] = (
    weekly_sales['Total_Weekly_Sales']
    .shift(1)
    .rolling(window=4)
    .mean()
)

weekly_sales['Rolling_Mean_8'] = (
    weekly_sales['Total_Weekly_Sales']
    .shift(1)
    .rolling(window=8)
    .mean()
)


# Remove null values created by lagging
weekly_sales = weekly_sales.dropna()


print("ML Features Created")



# ============================================
# 8. LOAD — Save Warehouse Tables
# ============================================

df.to_csv("cleaned_online_retail.csv", index=False)

weekly_sales.to_csv("weekly_sales.csv", index=False)


print("\nWarehouse Tables Saved:")
print("cleaned_online_retail.csv")
print("weekly_sales.csv")



# ============================================
# 9. FINAL SUMMARY
# ============================================

print("\nFinal Warehouse Shape:", weekly_sales.shape)

print("\nWarehouse Sample:")
print(weekly_sales.head())


print("\n===================================")
print("ETL PIPELINE COMPLETED SUCCESSFULLY")
print("===================================")
