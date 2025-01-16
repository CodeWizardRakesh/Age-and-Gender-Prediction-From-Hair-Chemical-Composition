import pandas as pd

# Load the Excel file
excel_file = "D:\projects\Age and Gender Prediction using Hair\Digitally Extracted dataset\SaAgeGen.xlsx"

# Read each sheet (if there are multiple sheets) or specify the sheet name
data = pd.read_excel(excel_file, sheet_name=None)  # Use None to read all sheets as a dictionary

# Iterate through the sheets and save each as CSV
for sheet_name, df in data.items():
    csv_file = f"{sheet_name}.csv"  # Name CSV file after the sheet name
    df.to_csv(csv_file, index=False)
    print(f"Converted {sheet_name} to {csv_file}")
