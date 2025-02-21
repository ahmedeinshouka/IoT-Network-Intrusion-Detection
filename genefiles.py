import pandas as pd
import numpy as np
import os

# Define the path for validation data
VALIDATION_DATA_PATH = "validation_data"

# Ensure the directory exists
os.makedirs(VALIDATION_DATA_PATH, exist_ok=True)

# Define a mapping for labels (Convert labels to numerical values)
LABEL_MAP = {
    "label": {"Benign": 0, "Malware": 1},
    "category": {"DoS": 0, "DDoS": 1, "PortScan": 2, "BruteForce": 3},
    "subcategory": {"HTTP_Flood": 0, "UDP_Flood": 1, "SYN_Flood": 2, "Dictionary_Attack": 3}
}

# List of required features
MODEL_FEATURES = [
    "Src_Port", "Dst_Port", "Protocol", "Flow_IAT_Mean", "Flow_IAT_Min",
    "Bwd_IAT_Mean", "Bwd_IAT_Min", "Pkt_Len_Max", "SYN_Flag_Cnt", "ACK_Flag_Cnt"
]

num_samples = 100  # Number of samples in the generated validation files

def generate_validation_data(filename, target_column):
    """
    Generates a validation CSV file with numerical labels.
    """
    if target_column not in LABEL_MAP:
        raise ValueError(f"Error: '{target_column}' not found in LABEL_MAP. Available keys: {list(LABEL_MAP.keys())}")

    df = pd.DataFrame({
        feature: np.random.randint(1, 100, size=num_samples) for feature in MODEL_FEATURES
    })

    # Assign numerical labels from LABEL_MAP
    df[target_column] = np.random.choice(list(LABEL_MAP[target_column].values()), size=num_samples)

    # Save to CSV
    file_path = os.path.join(VALIDATION_DATA_PATH, filename)
    df.to_csv(file_path, index=False)
    print(f"✅ Generated: {file_path}")

# Generate validation files with error checking
try:
    generate_validation_data("label_validation.csv", "label")
    generate_validation_data("category_validation.csv", "category")
    generate_validation_data("subcategory_validation.csv", "subcategory")
except ValueError as e:
    print(f"❌ {e}")
