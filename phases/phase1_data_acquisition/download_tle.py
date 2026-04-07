"""
TLE download and caching support for phase 1.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
from .data_acquisition import DataAcquisition


class TLEDownloader:
    """Download or synthesize TLE records for satellites."""

    def __init__(self, cache_dir: str = 'data/raw'):
        self.acquirer = DataAcquisition(cache_dir=cache_dir)

    def fetch(self, start_date: datetime, end_date: datetime, satellite_ids: Optional[List[str]] = None) -> pd.DataFrame:
        return self.acquirer.download_tle_data(start_date, end_date, satellite_ids)
