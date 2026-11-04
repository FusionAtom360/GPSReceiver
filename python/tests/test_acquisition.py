import math
import random
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from gpsrx.acquisition import FFTAcquisition, _sampled_code_fft
from gpsrx.ca_code import GPS_G2_TAPS


def independent_prn_code(prn: int) -> NDArray[np.float64]:
    """Reference LFSR implementation independent of the production generator."""
    tap1, tap2 = GPS_G2_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    chips: list[int] = []
    for _ in range(1023):
        chips.append(g1[9] ^ g2[tap1 - 1] ^ g2[tap2 - 1])
        g1 = [g1[2] ^ g1[9]] + g1[:9]
        g2 = [g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]] + g2[:9]
    return 2.0 * np.asarray(chips, dtype=np.float64) - 1.0


def make_signal(
    *,
    prn: int = 1,
    phase: int = 0,
    doppler: float = 0.0,
    sampling_rate: float = 1.023e6,
    milliseconds: int = 1,
    sample_offset: int = 0,
) -> NDArray[np.complex128]:
    samples_per_code = int(round(sampling_rate * 0.001))
    sample_numbers = np.arange(
        sample_offset,
        sample_offset + milliseconds * samples_per_code,
        dtype=np.float64,
    )
    chips = independent_prn_code(prn)
    chip_indices = np.floor(
        (sample_numbers + phase) * 1.023e6 / sampling_rate
    ).astype(np.int64) % 1023
    code = chips[chip_indices]
    carrier = np.exp(1j * 2.0 * np.pi * doppler * sample_numbers / sampling_rate)
    signal: NDArray[np.complex128] = code * carrier
    return signal


def test_sampled_code_fft_typical_matches_independent_fft() -> None:
    code = independent_prn_code(7)
    expected = np.conj(np.fft.fft(code))

    actual = _sampled_code_fft(7, 1.023e6)

    assert actual.shape == (1023,)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("sampling_rate", [1.023e6, 2.046e6, 4.092e6])
def test_sampled_code_fft_boundary_has_expected_sample_count(sampling_rate: float) -> None:
    expected_count = round(sampling_rate * 0.001)

    assert _sampled_code_fft(1, sampling_rate).shape == (expected_count,)


def test_sampled_code_fft_cache_reuses_same_array_for_same_key() -> None:
    _sampled_code_fft.cache_clear()

    first = _sampled_code_fft(1, 1.023e6)
    second = _sampled_code_fft(1, 1.023e6)

    assert first is second
    assert _sampled_code_fft.cache_info().hits == 1


@pytest.mark.parametrize("bad_prn", [0, 33, True, 1.5, "1"])
def test_acquisition_constructor_invalid_prn_rejected(bad_prn: object) -> None:
    expected = (
        TypeError
        if not isinstance(bad_prn, int) or isinstance(bad_prn, bool)
        else ValueError
    )

    with pytest.raises(expected):
        FFTAcquisition(bad_prn)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_rate",
    [0.0, -1.0, float("nan"), float("inf"), 0.1],
)
def test_acquisition_constructor_invalid_sampling_rate_rejected(bad_rate: float) -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1, sampling_rate=bad_rate)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_acquisition_constructor_invalid_threshold_rejected(bad_threshold: float) -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1, detection_threshold=bad_threshold)


@pytest.mark.parametrize("bad_ms", [0, -1, True, 1.5])
def test_acquisition_constructor_invalid_noncoherent_length_rejected(bad_ms: object) -> None:
    expected = TypeError if not isinstance(bad_ms, int) or isinstance(bad_ms, bool) else ValueError

    with pytest.raises(expected):
        FFTAcquisition(1, noncoherent_ms=bad_ms)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_exclusion", [-1.0, float("nan"), float("inf")])
def test_acquisition_constructor_invalid_exclusion_rejected(bad_exclusion: float) -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1, code_phase_exclusion_chips=bad_exclusion)


def test_acquisition_constructor_boundary_prns_are_accepted() -> None:
    assert FFTAcquisition(1).prn_number == 1
    assert FFTAcquisition(32).prn_number == 32


def test_acquire_typical_recovers_doppler_and_code_phase() -> None:
    acquisition = FFTAcquisition(1, detection_threshold=1.1)
    signal = make_signal(phase=100, doppler=1250.0)

    result = acquisition.acquire(signal, [-1250.0, 1250.0, 5000.0])

    assert result["detected"] is True
    assert result["prn_number"] == 1
    assert result["doppler_shift"] == pytest.approx(1250.0, abs=0.0)
    assert result["code_phase_samples"] == 100
    assert result["code_phase_chips"] == pytest.approx(100.0, abs=1e-12)
    assert result["peak_magnitude"] == pytest.approx(1023.0, abs=1e-9)


@pytest.mark.parametrize("phase", [0, 1, 1022])
def test_acquire_boundary_code_phase_wraparound_is_reported(phase: int) -> None:
    result = FFTAcquisition(1, detection_threshold=1.1).acquire(
        make_signal(phase=phase), [0.0]
    )

    assert result["detected"] is True
    assert result["code_phase_samples"] == phase


def test_acquire_uses_requested_sample_offset() -> None:
    offset = 37
    signal = np.concatenate(
        [np.zeros(offset, dtype=np.complex128), make_signal(phase=12)]
    )

    result = FFTAcquisition(1, detection_threshold=1.1).acquire(
        signal, [0.0], sample_offset=offset
    )

    assert result["code_phase_samples"] == 12


def test_acquire_boundary_zero_signal_is_not_detected() -> None:
    result = FFTAcquisition(1).acquire(np.zeros(1023, dtype=np.complex128), [0.0])

    assert result["detected"] is False
    assert result["peak_magnitude"] == 0.0
    assert result["doppler_shift"] == 0.0
    assert result["code_phase_samples"] == 0


def test_acquire_noncoherent_accumulation_detects_multiple_code_periods() -> None:
    signal = make_signal(phase=19, doppler=500.0, milliseconds=3)
    result = FFTAcquisition(1, noncoherent_ms=3, detection_threshold=1.1).acquire(
        signal, [500.0]
    )

    # Each 1 ms correlation has magnitude 1023; incoherent power accumulation
    # therefore preserves the peak magnitude sqrt(3 * 1023^2).
    assert result["detected"] is True
    assert result["code_phase_samples"] == 19
    assert result["peak_magnitude"] == pytest.approx(1023.0 * math.sqrt(3.0), abs=1e-9)


def test_acquire_noncoherent_accumulation_requires_all_requested_ms() -> None:
    signal = make_signal(milliseconds=1)

    with pytest.raises(ValueError):
        FFTAcquisition(1, noncoherent_ms=2).acquire(signal, [0.0])


def test_acquire_adversarial_wrong_prn_is_not_detected() -> None:
    signal = make_signal(prn=1, phase=20)

    result = FFTAcquisition(2).acquire(signal, [0.0])

    assert result["detected"] is False


@pytest.mark.parametrize("bad_bins", [[], [float("nan")], [float("inf")], [-float("inf")]])
def test_acquire_invalid_doppler_bins_rejected(bad_bins: Sequence[float]) -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1).acquire(np.zeros(1023, dtype=np.complex128), bad_bins)


@pytest.mark.parametrize("bad_signal", [np.zeros((2, 1023)), np.full(1023, np.nan + 0j)])
def test_acquire_invalid_signal_shape_or_values_rejected(bad_signal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1).acquire(bad_signal, [0.0])


def test_acquire_invalid_signal_type_rejected() -> None:
    with pytest.raises(TypeError):
        FFTAcquisition(1).acquire(["not", "numeric"], [0.0])


@pytest.mark.parametrize("bad_offset", [-1, True, 1.5])
def test_acquire_invalid_sample_offset_rejected(bad_offset: object) -> None:
    expected = (
        TypeError
        if not isinstance(bad_offset, int) or isinstance(bad_offset, bool)
        else ValueError
    )

    with pytest.raises(expected):
        FFTAcquisition(1).acquire(
            np.zeros(1023, dtype=np.complex128),
            [0.0],
            sample_offset=bad_offset,  # type: ignore[arg-type]
        )


def test_acquire_short_signal_rejected_at_minimum_boundary() -> None:
    with pytest.raises(ValueError):
        FFTAcquisition(1).acquire(np.zeros(1022, dtype=np.complex128), [0.0])


def test_acquire_randomized_doppler_selection_is_best_bin() -> None:
    rng = random.Random(20261031)
    for _ in range(10):
        doppler = rng.choice([-2000.0, -500.0, 0.0, 750.0, 2000.0])
        bins = [doppler - 500.0, doppler, doppler + 500.0]
        result = FFTAcquisition(1, detection_threshold=1.1).acquire(
            make_signal(doppler=doppler), bins
        )
        assert result["detected"] is True
        assert result["doppler_shift"] == doppler
