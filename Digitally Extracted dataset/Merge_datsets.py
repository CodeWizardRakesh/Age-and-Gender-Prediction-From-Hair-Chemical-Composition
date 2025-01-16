import pandas as pd

# Load all datasets
s_data = pd.read_csv("SAgeGen.csv")
ca_data = pd.read_csv("CaAgeGen.csv")
mg_data = pd.read_csv("MgAgeGen.csv")
zn_data = pd.read_csv("ZnAgeGen.csv")
cu_data = pd.read_csv("CuAgeGen.csv")

print("S_Data cols:",s_data.columns)
# Merge datasets one by one
merged_data = s_data.merge(ca_data, on=['Age', 'Gender'], how='outer')
merged_data = merged_data.merge(mg_data, on=['Age', 'Gender'], how='outer')
merged_data = merged_data.merge(zn_data, on=['Age', 'Gender'], how='outer')
merged_data = merged_data.merge(cu_data, on=['Age', 'Gender'], how='outer')

# Save the merged dataset
merged_data.to_csv("merged_dataset.csv", index=False)

print("Merged dataset saved as 'merged_dataset.csv'")
