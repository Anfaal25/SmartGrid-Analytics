"""
preprocess.py
-------------
Preprocesses the UCI Household Power dataset.

Steps:
1. Load raw CSV
2. Parse Date + Time into a datetime index
3. Convert numeric columns
4. Drop missing rows
5. Resample data to hourly means
6. Save final processed dataset

Output:
    data/processed_power.csv
"""

import pandas as pd
import numpy as np

def preprocess(input_path, output_path):
    print("Loading raw CSV...")

    df = pd.read_csv(
        input_path,
        sep=',',
        low_memory=False,
        na_values='?'
    )

    print("Converting Date + Time to datetime...")

    # Combine date and time
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    # Remove invalid datetimes
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')

    # Drop original string columns
    df = df.drop(columns=['Date', 'Time'])

    print("Converting numeric columns...")

    numeric_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Voltage',
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]

    # Convert all numeric columns properly
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Resampling to hourly averages...")
    df_hourly = df.resample('h').mean()

    print("Dropping rows with missing values after resampling...")
    df_hourly = df_hourly.dropna()

    print("Saving processed CSV...")
    df_hourly.to_csv(output_path)

    print(f"Preprocessing complete! Saved to: {output_path}")





if __name__ == "__main__":
    preprocess(
        "../data/household_power_consumption.csv",
        "../data/processed_power.csv"
    )
