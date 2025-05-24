import pandas as pd
import os

def load_data(filepath):
    """Load Excel data."""
    return pd.read_excel(filepath)

def extract_clean_data(df, threshold_db=-15.0):
    """
    Filter out rows that do not meet the valid valley depth threshold.
    Keep rows where Dip Value 1 < threshold_db.
    """
    valid_rows = df[df['Dip Value 1'] < threshold_db].copy()
    valid_rows = valid_rows.rename(columns={
        'Dip Freq 1': 'Resonant_Freq_GHz',
        'Dip Value 1': 'S21_dB'
    })
    return valid_rows

def save_cleaned_data(df, output_path='data/cleaned_valleys.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    filepath = 'data/C1024.xlsx'

    print("Loading data...")
    raw_df = load_data(filepath)

    print("Filtering valid resonances...")
    cleaned_df = extract_clean_data(raw_df)

    print(f"{len(cleaned_df)} valid samples retained.")
    print("Saving cleaned dataset...")
    save_cleaned_data(cleaned_df)
