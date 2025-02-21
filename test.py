import pandas as pd
import numpy as np

# List of features required by your model
MODEL_FEATURES = [
    'Src_Port', 'Dst_Port', 'Protocol', 'Flow_IAT_Mean', 'Flow_IAT_Min',
    'Bwd_IAT_Mean', 'Bwd_IAT_Min', 'Pkt_Len_Max', 'SYN_Flag_Cnt', 'ACK_Flag_Cnt'
]

# Generate synthetic data for testing
num_records = 1000  # Adjust the number of records as needed
data = {
    'Src_Port': np.random.randint(1024, 65535, num_records),
    'Dst_Port': np.random.randint(1024, 65535, num_records),
    'Protocol': np.random.choice([6, 17, 1], num_records),  # TCP (6), UDP (17), ICMP (1)
    'Flow_IAT_Mean': np.random.uniform(0, 1000, num_records),
    'Flow_IAT_Min': np.random.uniform(0, 500, num_records),
    'Bwd_IAT_Mean': np.random.uniform(0, 800, num_records),
    'Bwd_IAT_Min': np.random.uniform(0, 400, num_records),
    'Pkt_Len_Max': np.random.uniform(100, 1500, num_records),
    'SYN_Flag_Cnt': np.random.randint(0, 2, num_records),
    'ACK_Flag_Cnt': np.random.randint(0, 2, num_records)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
output_file = "synthetic_data.csv"
df.to_csv(output_file, index=False)
print(f"Synthetic dataset saved to {output_file}")
