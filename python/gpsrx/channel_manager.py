"""Channel manager for GPS receiver."""

import math
from dataclasses import dataclass
from typing import Protocol

from gpsrx.acquisition import FFTAcquisition
from gpsrx.source import IQSource
from gpsrx.tracking import GPSChannelTracker


class SampleSource(Protocol):
    sampling_rate: float

    def get_samples(self, num_samples: int) -> list[complex]:
        ...


@dataclass
class TrackedChannel:
    prn: int
    tracker: GPSChannelTracker


class ChannelManager:
    def __init__(self, signal_source: SampleSource) -> None:
        sampling_rate = signal_source.sampling_rate
        if isinstance(sampling_rate, bool) or not isinstance(
            sampling_rate, (int, float)
        ):
            raise TypeError("source.sampling_rate must be a real number")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("source.sampling_rate must be finite and positive")
        if not callable(getattr(signal_source, "get_samples", None)):
            raise TypeError("source must provide a callable get_samples method")
        self.channels: list[TrackedChannel] = []
        self.source = signal_source

    def acquire_all(self) -> dict[int, dict[str, float | int | bool]]:
        samples_per_ms = int(round(self.source.sampling_rate * 0.001))
        signal_block = self.source.get_samples(samples_per_ms)
        doppler_bins = range(-5000, 5001, 500)
        results_by_prn: dict[int, dict[str, float | int | bool]] = {}
        self.channels.clear()

        for prn in range(1, 33):
            acquisition = FFTAcquisition(
                prn,
                sampling_rate=self.source.sampling_rate,
            )
            results = acquisition.acquire(signal_block, doppler_bins)
            results_by_prn[prn] = results
            if results["detected"]:
                tracker = GPSChannelTracker(
                    prn,
                    self.source.sampling_rate,
                    results["doppler_shift"],
                    results["code_phase_chips"],
                )
                self.channels.append(TrackedChannel(prn=prn, tracker=tracker))
        return results_by_prn


if __name__ == "__main__":
    from gpsrx.source import IQSource

    with IQSource(4e6, "./data/captures/GPS-L1-2022-03-27.sigmf-data") as source:
        channel_manager = ChannelManager(source)
        results = channel_manager.acquire_all()
        for ms_index in range(15000):
            signal_block = source.get_samples(int(round(source.sampling_rate * 0.001)))
            for channel in channel_manager.channels:
                tracking_results = channel.tracker.process_millisecond(signal_block)
                if ms_index % 1000 == 0:
                    print(tracking_results)
