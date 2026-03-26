# experiment_tracker.py

"""
Experiment tracking utilities
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd


class ExperimentTracker:
    """
    Simple experiment tracker for logging metrics and configurations
    """

    def __init__(self, log_dir: str, config: Dict[str, Any], log_name: str):
        """
        Initialize experiment tracker

        Args:
            log_dir: Directory to save logs
            config: Experiment configuration
            log_name: Name for the log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{log_name}.csv"
        self.config_file = self.log_dir / f"{log_name}_config.json"

        # Save config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize log file
        self.logs = []

    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics for current step

        Args:
            metrics: Dictionary of metrics to log
        """
        self.logs.append(metrics)

        # Save to CSV
        if self.logs:
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_file, index=False)

    def close(self):
        """
        Close the tracker and save final logs
        """
        if self.logs:
            df = pd.DataFrame(self.logs)
            df.to_csv(self.log_file, index=False)
