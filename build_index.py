"""Build task_index.json from the consolidated parquet file.

The index maps each split to a list of row indices into the parquet file,
enabling O(1) lookups without loading the full dataset into memory.

Usage:
    python build_index.py
"""

import json

import pyarrow.parquet as pq

from constants import DATA_PATH

PARQUET_FILE = DATA_PATH / "nemotron_math_consolidated.parquet"
INDEX_PATH = DATA_PATH / "task_index.json"


def main():
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {PARQUET_FILE}\n"
            f"Run download_data.py first."
        )

    table = pq.read_table(str(PARQUET_FILE), columns=["split"])
    splits_col = table.column("split").to_pylist()

    splits: dict[str, list[int]] = {}
    for idx, split in enumerate(splits_col):
        splits.setdefault(split, []).append(idx)

    index = {"splits": splits}
    INDEX_PATH.write_text(json.dumps(index))

    print(f"Wrote {INDEX_PATH}")
    for name, indices in splits.items():
        print(f"  {name}: {len(indices):,} tasks")


if __name__ == "__main__":
    main()
