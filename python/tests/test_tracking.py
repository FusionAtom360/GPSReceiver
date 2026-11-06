import math
import random
import warnings

import numpy as np
import pytest

from gpsrx.ca_code import GPS_G2_TAPS
from gpsrx.tracking import GPSChannelTracker


def independent_bipolar_code(prn: int) -> np.ndarray:
    """Build the C/A code independently of GPSChannelTracker."""
    tap1, tap2 = GPS_G2_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    chips: list[int] = []
    for _ in range(1023):
        chips.append(g1[9] ^ g2[tap1 - 1] ^ g2[tap2 - 1])
        g1 = [g1[2] ^ g1[9]] + g1[:9]
        g2 = [g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]] + g2[:9]
    return 2.0 * np.asarray(chips, dtype=np.float64) - 1.0


def make_code_signal(
    prn: int,
    sampling_rate: float,
    code_phase: float = 0.0,
    sign: float = 1.0,
) -> np.ndarray:
    samples_per_code = int(round(sampling_rate * 0.001))
    sample_numbers = np.arange(samples_per_code, dtype=np.float64)
    chip_positions = (code_phase + sample_numbers * 1.023e6 / sampling_rate) % 1023.0
    return (
        sign * independent_bipolar_code(prn)[np.floor(chip_positions).astype(np.int64)]
    )


def test_tracker_constructor_typical_initializes_expected_state() -> None:
    tracker = GPSChannelTracker(7, 1.023e6, -1250.0, 12.5)

    assert tracker.prn_number == 7
    assert tracker.samples_per_code == 1023
    assert tracker.carrier_freq == -1250.0
    assert tracker.code_phase == 12.5
    assert tracker.epoch == 0
    assert tracker.last_demodulated_bit == 0


@pytest.mark.parametrize("prn", [1, 32])
def test_tracker_constructor_boundary_prns_are_accepted(prn: int) -> None:
    assert GPSChannelTracker(prn, 1.023e6, 0.0, 0.0).prn_number == prn


@pytest.mark.parametrize("bad_prn", [0, 33, True, 1.5, "1"])
def test_tracker_constructor_invalid_prn_is_rejected(bad_prn: object) -> None:
    expected = (
        TypeError
        if not isinstance(bad_prn, int) or isinstance(bad_prn, bool)
        else ValueError
    )
    with pytest.raises(expected):
        GPSChannelTracker(bad_prn, 1.023e6, 0.0, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_rate",
    [True, "1023000", 0.0, -1.0, 0.1, math.nan, math.inf],
)
def test_tracker_constructor_invalid_sampling_rate_is_rejected(
    bad_rate: object,
) -> None:
    expected = (
        TypeError
        if not isinstance(bad_rate, (int, float)) or isinstance(bad_rate, bool)
        else ValueError
    )
    with pytest.raises(expected):
        GPSChannelTracker(1, bad_rate, 0.0, 0.0)  # type: ignore[arg-type]


def test_tracker_constructor_minimum_sampling_rate_is_accepted() -> None:
    tracker = GPSChannelTracker(1, 1000.0, 0.0, 0.0)

    assert tracker.samples_per_code == 1


@pytest.mark.parametrize("bad_doppler", [True, "0", math.nan, math.inf, -math.inf])
def test_tracker_constructor_invalid_doppler_is_rejected(
    bad_doppler: object,
) -> None:
    expected = (
        TypeError
        if not isinstance(bad_doppler, (int, float)) or isinstance(bad_doppler, bool)
        else ValueError
    )
    with pytest.raises(expected):
        GPSChannelTracker(1, 1.023e6, bad_doppler, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_phase", [True, "0", math.nan, math.inf, -math.inf])
def test_tracker_constructor_invalid_code_phase_is_rejected(
    bad_phase: object,
) -> None:
    expected = (
        TypeError
        if not isinstance(bad_phase, (int, float)) or isinstance(bad_phase, bool)
        else ValueError
    )
    with pytest.raises(expected):
        GPSChannelTracker(1, 1.023e6, 0.0, bad_phase)  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", [-1023.25, 0.0, 1022.75, 1023.25, 2047.5])
def test_tracker_constructor_code_phase_is_normalized(phase: float) -> None:
    expected = phase % 1023.0

    tracker = GPSChannelTracker(1, 1.023e6, 0.0, phase)

    assert tracker.code_phase == expected


def test_tracker_local_codes_match_independent_early_prompt_late_replicas() -> None:
    sampling_rate = 4.0e6
    phase = 1022.75
    tracker = GPSChannelTracker(12, sampling_rate, 0.0, phase)
    sample_numbers = np.arange(tracker.samples_per_code, dtype=np.float64)
    offsets = sample_numbers * 1.023e6 / sampling_rate
    code = independent_bipolar_code(12)

    expected_early = code[np.floor((phase + 0.5 + offsets) % 1023.0).astype(np.int64)]
    expected_prompt = code[np.floor((phase + offsets) % 1023.0).astype(np.int64)]
    expected_late = code[np.floor((phase - 0.5 + offsets) % 1023.0).astype(np.int64)]

    early, prompt, late = tracker._generate_local_codes(phase)

    np.testing.assert_array_equal(early, expected_early)
    np.testing.assert_array_equal(prompt, expected_prompt)
    np.testing.assert_array_equal(late, expected_late)


def test_tracker_process_millisecond_zero_doppler_recovers_prompt_correlation() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)
    signal = make_code_signal(1, 1.023e6)

    metrics = tracker.process_millisecond(signal)

    # The independent prompt replica is +/-1 for all 1023 samples, so its
    # coherent real correlation is exactly 1023 and its power is 1023^2.
    assert metrics["prompt_i"] == pytest.approx(1023.0, abs=1e-10)
    assert metrics["prompt_q"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["prompt_power"] == pytest.approx(1023.0**2, abs=1e-8)
    assert metrics["carrier_phase_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["demodulated_bit"] == 0


def test_tracker_process_millisecond_uses_only_one_code_period() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)
    one_period = make_code_signal(1, 1.023e6)
    signal = np.concatenate([one_period, np.full(17, 7.0)])

    metrics = tracker.process_millisecond(signal)

    assert metrics["prompt_i"] == pytest.approx(1023.0, abs=1e-10)


def test_tracker_process_millisecond_empty_signal_is_rejected() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)

    with pytest.raises(ValueError, match="at least 1023"):
        tracker.process_millisecond(np.array([], dtype=np.complex128))


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(1022, dtype=np.complex128),
        np.zeros((1023, 1), dtype=np.complex128),
        np.full(1023, np.nan + 0j),
        np.full(1023, np.inf + 0j),
        ["not", "numeric"],
    ],
)
def test_tracker_process_millisecond_malformed_signal_is_rejected(
    signal: object,
) -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)
    expected = TypeError if isinstance(signal, list) else ValueError

    with pytest.raises(expected):
        tracker.process_millisecond(signal)  # type: ignore[arg-type]


def test_tracker_process_millisecond_zero_signal_keeps_state_finite() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)

    metrics = tracker.process_millisecond(np.zeros(1023, dtype=np.complex128))

    assert metrics["code_error"] == 0.0
    assert math.isfinite(tracker.code_phase)
    assert math.isfinite(tracker.carrier_freq)
    assert math.isfinite(metrics["carrier_phase_error"])


def test_tracker_process_millisecond_large_finite_signal_rejects_overflow() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="finite"):
            tracker.process_millisecond(
                np.full(1023, np.finfo(np.float64).max, dtype=np.complex128)
            )


def test_tracker_process_millisecond_advances_carrier_phase_at_block_boundary() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 1000.0, 0.0)
    expected_phase = (2.0 * np.pi * 1000.0 * 0.001) % (2.0 * np.pi)

    tracker.process_millisecond(np.zeros(1023, dtype=np.complex128))

    assert tracker.carrier_phase == pytest.approx(expected_phase, abs=1e-12)


def test_tracker_process_millisecond_navigation_bit_emits_after_20_blocks() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)
    positive_signal = make_code_signal(1, 1.023e6)

    outputs = [
        tracker.process_millisecond(positive_signal)["demodulated_bit"]
        for _ in range(20)
    ]

    assert outputs[:19] == [0] * 19
    assert outputs[19] == 1


def test_tracker_process_millisecond_navigation_bit_integrates_signs() -> None:
    tracker = GPSChannelTracker(1, 1.023e6, 0.0, 0.0)
    positive_signal = make_code_signal(1, 1.023e6)

    outputs = []
    for index in range(20):
        sign = 1.0 if index < 11 else -1.0
        outputs.append(tracker.process_millisecond(sign * positive_signal)["prompt_i"])

    assert sum(outputs) > 0.0
    assert tracker.last_demodulated_bit == 1


def test_tracker_process_millisecond_random_code_phase_stays_in_range() -> None:
    rng = random.Random(20261104)
    tracker = GPSChannelTracker(7, 1.023e6, 0.0, 0.0)
    signal = make_code_signal(7, 1.023e6)

    for _ in range(25):
        tracker.code_phase = rng.uniform(-4092.0, 4092.0)
        tracker.process_millisecond(signal)
        assert 0.0 <= tracker.code_phase < 1023.0


def test_tracker_local_codes_random_phases_are_binary_and_period_sized() -> None:
    rng = random.Random(20261104)
    tracker = GPSChannelTracker(5, 2.046e6, 0.0, 0.0)

    for _ in range(20):
        phase = rng.uniform(-4092.0, 4092.0)
        early, prompt, late = tracker._generate_local_codes(phase)

        assert early.shape == (2046,)
        assert prompt.shape == (2046,)
        assert late.shape == (2046,)
        assert set(np.unique(early)) <= {-1.0, 1.0}
        assert set(np.unique(prompt)) <= {-1.0, 1.0}
        assert set(np.unique(late)) <= {-1.0, 1.0}
