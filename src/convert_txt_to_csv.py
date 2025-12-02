"""
convert_txt_to_csv.py
Converts the raw UCI electric power consumption TXT file into a CSV.
No preprocessing is done here.
"""

import pandas as pd

def convert_txt_to_csv(input_path, output_path):
    print("📥 Loading raw TXT file (this may take a minute)...")

    df = pd.read_csv(
        input_path,
        sep=';',
        low_memory=False,
        na_values='?',
        dtype=str  # keep everything as strings for the raw version
    )

    print("📤 Saving as CSV...")
    df.to_csv(output_path, index=False)

    print(f"✅ Conversion complete! CSV saved at: {output_path}")


if __name__ == "__main__":
    convert_txt_to_csv(
        "../data/household_power_consumption.txt",
        "../data/household_power_consumption.csv"
    )
