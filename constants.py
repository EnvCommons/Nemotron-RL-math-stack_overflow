"""Path management for Nemotron-RL-Math-Stack-Overflow environment."""

from pathlib import Path

# Automatically detect environment and set data path
if Path("/orwd_data").exists():
    # Production: data mounted at /orwd_data
    DATA_PATH = Path("/orwd_data")
else:
    # Local development: data in environment directory
    DATA_PATH = Path(__file__).parent
