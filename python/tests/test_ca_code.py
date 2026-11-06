import pytest

from gpsrx.ca_code import GPS_G2_TAPS, GPSPRNGenerator


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
        g2 = [g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]] + g2[:9]
    return code


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
