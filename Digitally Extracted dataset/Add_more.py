import pandas as pd
import numpy as np

# Existing correlation matrix
correlation_matrix = np.array([
    [1.0, 0.368643, 0.366470, 0.312813, -0.243420, -0.296814],
    [0.368643, 1.0, 0.5, 0.4, -0.2, -0.3],
    [0.366470, 0.5, 1.0, 0.5, -0.3, -0.4],
    [0.312813, 0.4, 0.5, 1.0, -0.1, -0.2],
    [-0.243420, -0.2, -0.3, -0.1, 1.0, 0.4],
    [-0.296814, -0.3, -0.4, -0.2, 0.4, 1.0]
])

# Mean and standard deviation for features
means = [40, 900, 100, 150, 38, 30]  # Example means for Age, Mg, Ca, Zn, S, Cu
std_devs = [25, 200, 30, 20, 5, 10]  # Example standard deviations

# Generate synthetic data
num_samples = 50  # Number of new records
synthetic_data = np.random.multivariate_normal(means, correlation_matrix * np.outer(std_devs, std_devs), size=num_samples)

# Convert to DataFrame
columns = ['Age', 'Mg', 'Ca', 'Zn', 'S', 'Cu']
synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

# Round and adjust ranges to match realistic values
synthetic_df['Age'] = synthetic_df['Age'].clip(2, 90).round(0)
synthetic_df['Mg'] = synthetic_df['Mg'].clip(400, 1500).round(0)
synthetic_df['Ca'] = synthetic_df['Ca'].clip(50, 150).round(0)
synthetic_df['Zn'] = synthetic_df['Zn'].clip(70, 200).round(0)
synthetic_df['S'] = synthetic_df['S'].clip(15, 50).round(2)
synthetic_df['Cu'] = synthetic_df['Cu'].clip(10, 50).round(1)

# Combine with original dataset
original_data = pd.read_csv("D:\projects\Age and Gender Prediction using Hair\Digitally Extracted dataset\Extra\Extra.csv") # Assuming the original data is copied to the clipboard
combined_data = pd.concat([original_data, synthetic_df], ignore_index=True)

# Validate correlations
print(combined_data.corr())

# Save or view the new dataset
print(combined_data)
pd.to_csv("Hair_extra.csv")