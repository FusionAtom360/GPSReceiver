"""Receiver configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReceiverConfig:
    """Configuration shared by source, acquisition, and tracking stages."""

    sample_rate: float = 2_046_000.0
    intermediate_frequency: float = 0.0
    sample_format: str = "complex64"
    doppler_search_hz: float = 5_000.0
    doppler_step_hz: float = 500.0
    integration_ms: int = 1


def load_config(path: Path) -> ReceiverConfig:
    """Load a YAML configuration file."""
    values: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ReceiverConfig(
        sample_rate=float(values.get("sample_rate", ReceiverConfig.sample_rate)),
        intermediate_frequency=float(
            values.get("intermediate_frequency", ReceiverConfig.intermediate_frequency)
        ),
        sample_format=str(values.get("sample_format", ReceiverConfig.sample_format)),
        doppler_search_hz=float(
            values.get("doppler_search_hz", ReceiverConfig.doppler_search_hz)
        ),
        doppler_step_hz=float(
            values.get("doppler_step_hz", ReceiverConfig.doppler_step_hz)
        ),
        integration_ms=int(values.get("integration_ms", ReceiverConfig.integration_ms)),
    )
