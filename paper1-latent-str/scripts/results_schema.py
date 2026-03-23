#!/usr/bin/env python3
"""
Results YAML schema for paper injection.

Defines the structure of results.yaml files that feed into inject_results.py.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
import yaml


@dataclass
class ResultsSchema:
    """Schema for experiment results YAML."""

    # Metadata
    experiment_id: str
    generated_at: str
    config_path: str

    # Inline metrics (for {{metric_name}} placeholders)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Full tables (for {{TABLE:table_name}} placeholders)
    tables: Dict[str, str] = field(default_factory=dict)

    def to_yaml(self, path: str) -> None:
        """Save results to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'ResultsSchema':
        """Load results from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


def create_empty_results(experiment_id: str, config_path: str) -> ResultsSchema:
    """Create empty results schema with metadata."""
    return ResultsSchema(
        experiment_id=experiment_id,
        generated_at=datetime.now().isoformat(),
        config_path=config_path,
        metrics={},
        tables={},
    )
