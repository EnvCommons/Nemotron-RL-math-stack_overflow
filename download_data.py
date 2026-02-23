"""Dataset setup instructions for Nemotron-RL-Math-Stack-Overflow.

This environment requires a pre-consolidated parquet file containing the dataset.

IMPORTANT: This environment does NOT use the HuggingFace datasets library.
"""

import pandas as pd
from pathlib import Path

from constants import DATA_PATH


def verify_dataset():
    """Verify that the dataset file exists and is valid."""

    parquet_file = DATA_PATH / "nemotron_math_consolidated.parquet"

    if not parquet_file.exists():
        print(f"ERROR: Dataset file not found at {parquet_file}")
        print("\nTo obtain the dataset:")
        print("1. Download the consolidated parquet file from your data source")
        print("2. Place it at: nemotron_math_consolidated.parquet")
        print("3. Or mount it at: /orwd_data/nemotron_math_consolidated.parquet (production)")
        return False

    print(f"Found dataset file: {parquet_file}")
    print(f"File size: {parquet_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Verify structure
    try:
        df = pd.read_parquet(parquet_file)
        print(f"\nDataset Statistics:")
        print(f"  Total tasks: {len(df):,}")
        print(f"  Train split: {len(df[df['split'] == 'train']):,}")
        print(f"  Validation split: {len(df[df['split'] == 'validation']):,}")
        print(f"  Columns: {list(df.columns)}")

        required_columns = ['task_id', 'split', 'question', 'expected_answer', 'row_idx']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"\nERROR: Missing required columns: {missing}")
            return False

        print("\n✓ Dataset is valid and ready to use!")
        return True

    except Exception as e:
        print(f"\nERROR: Failed to read dataset: {e}")
        return False


if __name__ == "__main__":
    verify_dataset()
