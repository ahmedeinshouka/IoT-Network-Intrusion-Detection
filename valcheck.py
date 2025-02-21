import pandas as pd

validation_files = {
    'label': 'validation_data/label_validation.csv',
    'category': 'validation_data/category_validation.csv',
    'subcategory': 'validation_data/subcategory_validation.csv'
}

for key, path in validation_files.items():
    df = pd.read_csv(path)
    print(f"\n🔍 Checking '{path}'")
    print("Columns:", df.columns.tolist())  # Check available columns
    print("First rows:", df.head())  # View sample data
