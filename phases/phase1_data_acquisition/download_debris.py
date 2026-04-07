"""
Debris catalog download helpers for phase 1.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np


class DebrisCatalogDownloader:
    """Provide synthetic debris catalog entries for collision analysis."""

    def __init__(self, cache_dir: str = 'data/raw'):
        self.cache_dir = cache_dir

    def fetch(self, count: int = 500, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        entries = []
        for i in range(count):
            entries.append({
                'debris_id': f'DEBRIS-{i:04d}',
                'epoch': datetime.utcnow(),
                'object_type': np.random.choice(['fragment', 'rocket_body', 'satellite']),
                'cross_section_m2': float(np.random.uniform(0.1, 10.0)),
                'mass_kg': float(np.random.uniform(0.01, 100.0)),
                'altitude_km': float(np.random.uniform(200, 1200)),
            })
        df = pd.DataFrame(entries)
        return df
