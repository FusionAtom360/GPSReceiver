import math
import random

import pytest

from gpsrx.synthetic import GPS_G2_TAPS, GPSPRNGenerator, SyntheticGPSChannel


class ZeroRNG:
    def gauss(self, _mu: float, _sigma: float) -> float:
        return 0.0


def independent_prn_code(prn: int) -> list[int]:
    """Reference LFSR calculation, independent of GPSPRNGenerator's object state."""
    tap1, tap2 = GPS_G2_TAPS[prn]
    g1 = [1] * 10
    g2 = [1] * 10
    code = []
    for _ in range(1023):
        output = g1[9] ^ g2[tap1 - 1] ^ g2[tap2 - 1]
        code.append(output)
        g1 = [g1[2] ^ g1[9]] + g1[:9]
        g2 = [
            g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        ] + g2[:9]
    return code


def make_channel(
    *,
    prn: int = 1,
    doppler: float = 0.0,
    code_phase: float = 0.0,
    cn0: float = 0.0,
    sampling_rate: float = 1.023e6,
    navigation_data: list[int] | None = None,
    rng: object | None = None,
) -> SyntheticGPSChannel:
    return SyntheticGPSChannel(
        [prn],
        [doppler],
        [code_phase],
        [cn0],
        sampling_rate=sampling_rate,
        navigation_data=navigation_data,
        rng=rng,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("prn", [1, 2, 16, 32])
def test_prn_generate_code_typical_prns_match_independent_reference(prn: int) -> None:
    assert GPSPRNGenerator(prn).generate_code() == independent_prn_code(prn)


def test_prn_generate_code_boundary_has_1023_binary_chips_and_repeats() -> None:
    generator = GPSPRNGenerator(1)

    first_period = generator.generate_code()
    second_period = generator.generate_code()

    assert len(first_period) == 1023
    assert set(first_period) == {0, 1}
    assert second_period == first_period


@pytest.mark.parametrize("bad_prn", [0, -1, 33, 1.0, True, "1"])
def test_prn_constructor_invalid_prn_rejected(bad_prn: object) -> None:
    expected = (
        TypeError
        if not isinstance(bad_prn, int) or isinstance(bad_prn, bool)
        else ValueError
    )
    with pytest.raises(expected):
        GPSPRNGenerator(bad_prn)  # type: ignore[arg-type]


def test_prn_clock_advances_state_by_one_chip() -> None:
    generator = GPSPRNGenerator(1)
    expected = independent_prn_code(1)

    assert [generator.clock() for _ in range(4)] == expected[:4]


def test_channel_generate_samples_typical_matches_code_and_amplitude() -> None:
    samples = make_channel(rng=ZeroRNG()).generate_baseband_samples(3)
    reference_code = independent_prn_code(1)
    amplitude = math.sqrt(1.0 / 1.023e6)

    expected = [complex(amplitude if chip else -amplitude, 0.0) for chip in reference_code[:3]]
    assert samples == pytest.approx(expected, abs=1e-15)


def test_channel_generate_samples_full_code_period_matches_reference() -> None:
    samples = make_channel(rng=ZeroRNG()).generate_baseband_samples(1023)
    amplitude = math.sqrt(1.0 / 1.023e6)
    expected = [
        complex(amplitude if chip else -amplitude, 0.0)
        for chip in independent_prn_code(1)
    ]

    assert samples == pytest.approx(expected, abs=1e-15)


def test_channel_generate_samples_boundary_zero_length_returns_empty_without_advancing() -> None:
    channel = make_channel(rng=ZeroRNG())

    assert channel.generate_baseband_samples(0) == []
    assert channel.sample_index == 0


def test_channel_generate_samples_boundary_wraps_code_phase() -> None:
    wrapped = make_channel(code_phase=1023.0, rng=ZeroRNG()).generate_baseband_samples(4)
    zero = make_channel(code_phase=0.0, rng=ZeroRNG()).generate_baseband_samples(4)

    assert wrapped == zero


@pytest.mark.parametrize("bad_count", [-1, 1.5, True, "1"])
def test_channel_generate_samples_invalid_count_rejected(bad_count: object) -> None:
    expected = (
        ValueError
        if isinstance(bad_count, int) and not isinstance(bad_count, bool)
        else TypeError
    )
    with pytest.raises(expected):
        make_channel(rng=ZeroRNG()).generate_baseband_samples(bad_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_rate", 0.0),
        ("sampling_rate", float("nan")),
        ("doppler", 5000.1),
        ("doppler", float("inf")),
        ("code_phase", float("nan")),
        ("cn0", -0.1),
        ("cn0", 50.1),
        ("cn0", float("inf")),
    ],
)
def test_channel_constructor_invalid_numeric_input_rejected(field: str, value: float) -> None:
    values = {"doppler": 0.0, "code_phase": 0.0, "cn0": 0.0, "sampling_rate": 1.023e6}
    values[field] = value
    with pytest.raises(ValueError):
        make_channel(
            doppler=values["doppler"],
            code_phase=values["code_phase"],
            cn0=values["cn0"],
            sampling_rate=values["sampling_rate"],
        )


def test_channel_constructor_mismatched_satellite_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        SyntheticGPSChannel([1, 2], [0.0], [0.0, 0.0], [0.0, 0.0])


@pytest.mark.parametrize("bad_navigation_data", [[], [0], [1, -1, 2], [True]])
def test_channel_constructor_invalid_navigation_data_rejected(
    bad_navigation_data: list[int],
) -> None:
    with pytest.raises(ValueError):
        make_channel(navigation_data=bad_navigation_data)


def test_channel_generate_samples_navigation_bit_changes_after_twenty_code_periods() -> None:
    samples = make_channel(
        cn0=0.0,
        navigation_data=[1, -1],
        rng=ZeroRNG(),
    ).generate_baseband_samples(20 * 1023 + 1)
    amplitude = math.sqrt(1.0 / 1.023e6)
    code_first_chip = 1 if independent_prn_code(1)[0] else -1

    assert samples[0] == pytest.approx(complex(code_first_chip * amplitude), abs=1e-15)
    assert samples[20 * 1023] == pytest.approx(complex(-code_first_chip * amplitude), abs=1e-15)


def test_channel_generate_samples_continuity_matches_one_shot_generation() -> None:
    split = make_channel(doppler=2000.0, code_phase=12.25, rng=random.Random(1234))
    one_shot = make_channel(doppler=2000.0, code_phase=12.25, rng=random.Random(1234))

    split_samples = split.generate_baseband_samples(17) + split.generate_baseband_samples(31)
    one_shot_samples = one_shot.generate_baseband_samples(48)

    assert split_samples == pytest.approx(one_shot_samples, abs=1e-15)


def test_channel_generate_samples_seeded_rng_is_reproducible() -> None:
    first = make_channel(rng=random.Random(55)).generate_baseband_samples(32)
    second = make_channel(rng=random.Random(55)).generate_baseband_samples(32)

    assert first == second


def test_channel_generate_samples_noise_only_empty_satellite_set_returns_finite_samples() -> None:
    samples = SyntheticGPSChannel(
        [], [], [], [], rng=random.Random(7)
    ).generate_baseband_samples(32)

    assert len(samples) == 32
    assert all(math.isfinite(sample.real) and math.isfinite(sample.imag) for sample in samples)


def test_channel_phase_property_randomized_fixed_seed_is_periodic_in_code_phase() -> None:
    randomizer = random.Random(20261030)
    for _ in range(20):
        phase = randomizer.uniform(-10_000.0, 10_000.0)
        first = make_channel(code_phase=phase, rng=ZeroRNG()).generate_baseband_samples(8)
        second = make_channel(code_phase=phase + 1023.0, rng=ZeroRNG()).generate_baseband_samples(8)
        assert first == second
