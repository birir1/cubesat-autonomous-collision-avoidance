"""
Preprocessing utilities for raw phase 1 data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class DataPreprocessor:
    """Clean and transform raw satellite and conjunction data."""

    def __init__(self, raw_dir: str = 'data/raw', processed_dir: str = 'data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_tle(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        df = pd.read_csv(path)
        df['epoch'] = pd.to_datetime(df['epoch'], errors='coerce')
        df = df.dropna(subset=['epoch'])
        df['inclination'] = pd.to_numeric(df['inclination'], errors='coerce').fillna(0.0)
        df['mean_motion'] = pd.to_numeric(df['mean_motion'], errors='coerce').fillna(0.0)
        out_path = self.processed_dir / f'processed_{filename}'
        df.to_csv(out_path, index=False)
        return df

    def preprocess_conjunctions(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        df = pd.read_csv(path)
        df['tca'] = pd.to_datetime(df['tca'], errors='coerce')
        df = df.dropna(subset=['tca'])
        df['collision_probability'] = pd.to_numeric(df['collision_probability'], errors='coerce').fillna(0.0)
        out_path = self.processed_dir / f'processed_{filename}'
        df.to_csv(out_path, index=False)
        return df

    def preprocess_radar(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['range'] = pd.to_numeric(df['range'], errors='coerce').fillna(0.0)
        out_path = self.processed_dir / f'processed_{filename}'
        df.to_csv(out_path, index=False)
        return df
