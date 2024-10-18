import pandas as pd

def load_data(filepath):
    """Load product data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Basic data preprocessing, e.g., handling missing values."""
    df = df.dropna(subset=['description'])  # Ensure all products have descriptions
    return df
