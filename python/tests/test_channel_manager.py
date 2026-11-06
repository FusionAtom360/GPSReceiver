import math
import random
from dataclasses import dataclass
from typing import Any

import pytest

import gpsrx.channel_manager as channel_manager_module
from gpsrx.channel_manager import ChannelManager, TrackedChannel


@dataclass
class FakeSource:
    sampling_rate: float
    samples: list[complex]

    def __post_init__(self) -> None:
        self.requests: list[int] = []

    def get_samples(self, num_samples: int) -> list[complex]:
        self.requests.append(num_samples)
        return self.samples


class FakeAcquisition:
    instances: list["FakeAcquisition"] = []
    detected_prns: set[int] = set()

    def __init__(self, prn_number: int, sampling_rate: float) -> None:
        self.prn_number = prn_number
        self.sampling_rate = sampling_rate
        self.calls: list[tuple[list[complex], tuple[int, ...]]] = []
        type(self).instances.append(self)

    def acquire(
        self, signal_block: list[complex], doppler_bins: range
    ) -> dict[str, float | int | bool]:
        bins = tuple(doppler_bins)
        self.calls.append((signal_block, bins))
        detected = self.prn_number in type(self).detected_prns
        return {
            "prn_number": self.prn_number,
            "detected": detected,
            "peak_magnitude": 10.0 if detected else 0.0,
            "peak_metric": 5.0 if detected else 1.0,
            "doppler_shift": 1250.0 if detected else 0.0,
            "code_phase_samples": 777 if detected else 0,
            "code_phase_chips": 12.25 if detected else 0.0,
        }


class FakeTracker:
    instances: list["FakeTracker"] = []

    def __init__(
        self,
        prn_number: int,
        sampling_rate: float,
        initial_doppler: float,
        initial_code_phase: float,
    ) -> None:
        self.arguments = (
            prn_number,
            sampling_rate,
            initial_doppler,
            initial_code_phase,
        )
        type(self).instances.append(self)


@pytest.fixture
def fake_acquisition_and_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAcquisition.instances = []
    FakeAcquisition.detected_prns = set()
    FakeTracker.instances = []
    monkeypatch.setattr(channel_manager_module, "FFTAcquisition", FakeAcquisition)
    monkeypatch.setattr(channel_manager_module, "GPSChannelTracker", FakeTracker)


def test_channel_manager_init_typical_source_starts_empty() -> None:
    source = FakeSource(2_046_000.0, [])

    manager = ChannelManager(source)

    assert manager.source is source
    assert manager.channels == []


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_channel_manager_init_nonpositive_or_nonfinite_rate_rejected(
    sampling_rate: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ChannelManager(FakeSource(sampling_rate, []))


@pytest.mark.parametrize("sampling_rate", [True, "2046000", None])
def test_channel_manager_init_non_numeric_rate_rejected(
    sampling_rate: object,
) -> None:
    with pytest.raises(TypeError, match="real number"):
        ChannelManager(FakeSource(sampling_rate, []))  # type: ignore[arg-type]


def test_channel_manager_init_source_without_get_samples_rejected() -> None:
    class MissingSamples:
        sampling_rate = 2_046_000.0

    with pytest.raises(TypeError, match="get_samples"):
        ChannelManager(MissingSamples())  # type: ignore[arg-type]


def test_channel_manager_acquire_all_typical_returns_all_prn_results(
    fake_acquisition_and_tracker: None,
) -> None:
    source = FakeSource(2_046_000.0, [1.0 + 2.0j])
    FakeAcquisition.detected_prns = {1, 32}

    results = ChannelManager(source).acquire_all()

    assert list(results) == list(range(1, 33))
    assert results[1]["detected"] is True
    assert results[32]["detected"] is True
    assert results[2]["detected"] is False
    assert source.requests == [2046]


def test_channel_manager_acquire_all_uses_expected_search_grid(
    fake_acquisition_and_tracker: None,
) -> None:
    source = FakeSource(1_023_000.0, [])

    ChannelManager(source).acquire_all()

    expected_bins = tuple(range(-5000, 5001, 500))
    assert len(FakeAcquisition.instances) == 32
    assert all(
        instance.sampling_rate == 1_023_000.0 for instance in FakeAcquisition.instances
    )
    assert all(
        instance.calls[0][1] == expected_bins for instance in FakeAcquisition.instances
    )


def test_channel_manager_acquire_all_passes_code_phase_chips_to_tracker(
    fake_acquisition_and_tracker: None,
) -> None:
    source = FakeSource(4_000_000.0, [])
    FakeAcquisition.detected_prns = {12}

    manager = ChannelManager(source)
    manager.acquire_all()

    # Acquisition reports both units; the tracker constructor requires chips.
    assert FakeTracker.instances[0].arguments == (12, 4_000_000.0, 1250.0, 12.25)
    assert manager.channels == [
        TrackedChannel(
            prn=12,
            tracker=manager.channels[0].tracker,
            navigation=manager.channels[0].navigation,
        )
    ]


def test_channel_manager_acquire_all_repeated_call_replaces_channels(
    fake_acquisition_and_tracker: None,
) -> None:
    source = FakeSource(1_023_000.0, [])
    FakeAcquisition.detected_prns = {1, 32}
    manager = ChannelManager(source)

    manager.acquire_all()
    first_trackers = [channel.tracker for channel in manager.channels]
    manager.acquire_all()

    assert len(manager.channels) == 2
    assert [channel.prn for channel in manager.channels] == [1, 32]
    assert [channel.tracker for channel in manager.channels] != first_trackers


def test_channel_manager_acquire_all_propagates_empty_source_failure() -> None:
    source = FakeSource(1_023_000.0, [])

    with pytest.raises(ValueError, match="at least 1023 samples"):
        ChannelManager(source).acquire_all()


def test_channel_manager_acquire_all_uses_rounding_for_one_ms_sample_count(
    fake_acquisition_and_tracker: None,
) -> None:
    source = FakeSource(1_023_500.0, [])

    ChannelManager(source).acquire_all()

    # 1 ms contains 1023.5 samples; Python round uses bankers rounding to 1024.
    assert source.requests == [1024]


def test_channel_manager_acquire_all_sample_count_property_for_seeded_rates(
    fake_acquisition_and_tracker: None,
) -> None:
    rng = random.Random(20261104)
    rates = [rng.randint(1_000, 8_000_000) for _ in range(10)]

    for rate in rates:
        source = FakeSource(float(rate), [])
        ChannelManager(source).acquire_all()

        # Derivation: one millisecond has rate / 1000 samples, rounded to an int.
        assert source.requests == [round(rate * 0.001)]


def test_channel_manager_acquire_all_does_not_accept_malformed_acquisition_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedAcquisition:
        def __init__(self, prn_number: int, sampling_rate: float) -> None:
            pass

        def acquire(self, signal_block: Any, doppler_bins: Any) -> dict[str, object]:
            return {"detected": "yes"}

    monkeypatch.setattr(channel_manager_module, "FFTAcquisition", MalformedAcquisition)

    with pytest.raises(KeyError):
        ChannelManager(FakeSource(1_023_000.0, [])).acquire_all()
