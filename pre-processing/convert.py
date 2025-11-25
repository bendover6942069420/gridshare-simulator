import pandas as pd

# Read the semicolon-delimited TXT file
# Assumes first row has headers: Date;Time;Global_active_power; etc.
df = pd.read_csv('household_power_consumption.txt', delimiter=';', header=0, na_values=['?'])  # '?' for missing data in this dataset

# Optional: Preview the data (first 5 rows)
print(df.head())

# Write to comma-separated CSV
df.to_csv('output.csv', index=False)  # No row indices

print("Conversion complete! Columns:", df.columns.tolist())
print("Output saved as output.csv")