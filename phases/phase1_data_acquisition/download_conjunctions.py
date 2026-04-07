"""
Conjunction download support for phase 1.
"""

from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import numpy as np


class ConjunctionDownloader:
    """Fetch or synthesize conjunction warnings for satellite pairs."""

    def fetch(self, start_date: datetime, end_date: datetime, min_risk: float = 1e-6, count: int = 200) -> pd.DataFrame:
        conjunctions = []
        window = max(1, int((end_date - start_date).days))
        for i in range(count):
            tca = start_date + timedelta(days=np.random.uniform(0, window))
            conjunctions.append({
                'primary_satellite': f'SAT-{np.random.randint(1000, 9999)}',
                'secondary_satellite': f'SAT-{np.random.randint(10000, 19999)}',
                'tca': tca,
                'miss_distance_km': float(np.random.uniform(0.1, 5.0)),
                'collision_probability': float(np.random.uniform(min_risk, 1e-3)),
                'relative_velocity_kms': float(np.random.uniform(0.5, 15.0))
            })
        df = pd.DataFrame(conjunctions)
        df['tca'] = pd.to_datetime(df['tca'])
        return df
