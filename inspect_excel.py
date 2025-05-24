import pandas as pd

# Load the Excel file
filepath = 'data/C1024.xlsx'  # Adjust path as needed
df = pd.read_excel(filepath)

# Print basic info
print("Column names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())
