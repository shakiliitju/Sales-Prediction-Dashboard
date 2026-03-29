# ============================================
# DATA WAREHOUSE LOADING SCRIPT
# ============================================

import pandas as pd

print("===================================")
print("LOADING DATA INTO DATA WAREHOUSE")
print("===================================")


# ============================================
# 1. Load Cleaned Data (From ETL)
# ============================================

df = pd.read_csv("cleaned_online_retail.csv")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("\nCleaned data loaded successfully")
print("Shape:", df.shape)



# ============================================
# 2. Create FACT TABLE (Weekly Sales)
# ============================================

fact_weekly_sales = df.resample('W-MON', on='InvoiceDate').agg({

    'Revenue': 'sum',
    'InvoiceNo': 'count',
    'CustomerID': 'nunique',
    'Quantity': 'sum',
    'StockCode': 'nunique'

}).reset_index()


fact_weekly_sales.rename(columns={

    'Revenue': 'Total_Weekly_Sales',
    'InvoiceNo': 'Transaction_Count',
    'CustomerID': 'Customer_Count',
    'Quantity': 'Quantity_Sold',
    'StockCode': 'Unique_Products_Sold'

}, inplace=True)


print("\nFACT TABLE Created")



# ============================================
# 3. Create BUSINESS INTELLIGENCE METRICS
# ============================================

fact_weekly_sales['Avg_Order_Value'] = (
    fact_weekly_sales['Total_Weekly_Sales']
    / fact_weekly_sales['Transaction_Count']
)

fact_weekly_sales['Avg_Quantity_Per_Transaction'] = (
    fact_weekly_sales['Quantity_Sold']
    / fact_weekly_sales['Transaction_Count']
)


print("BI Metrics Created")



# ============================================
# 4. Save Warehouse Tables
# ============================================

fact_weekly_sales.to_csv("weekly_sales.csv", index=False)


print("\nData Warehouse Tables Saved:")
print("weekly_sales.csv")



# ============================================
# 5. Warehouse Summary
# ============================================

print("\nWarehouse Shape:", fact_weekly_sales.shape)

print("\nSample Data:")
print(fact_weekly_sales.head())


print("\n===================================")
print("DATA WAREHOUSE LOADED SUCCESSFULLY")
print("===================================")
