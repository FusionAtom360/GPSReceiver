import random

import pytest

from gpsrx.navigation import Navigation


def reference_parity(
    data: list[int], previous_d29: int, previous_d30: int
) -> list[int]:
    d = data
    return [
        previous_d29
        ^ d[0]
        ^ d[1]
        ^ d[2]
        ^ d[4]
        ^ d[5]
        ^ d[9]
        ^ d[10]
        ^ d[11]
        ^ d[12]
        ^ d[13]
        ^ d[16]
        ^ d[17]
        ^ d[19]
        ^ d[22],
        previous_d30
        ^ d[1]
        ^ d[2]
        ^ d[3]
        ^ d[5]
        ^ d[6]
        ^ d[10]
        ^ d[11]
        ^ d[12]
        ^ d[13]
        ^ d[14]
        ^ d[17]
        ^ d[18]
        ^ d[20]
        ^ d[23],
        previous_d29
        ^ d[0]
        ^ d[2]
        ^ d[3]
        ^ d[4]
        ^ d[6]
        ^ d[7]
        ^ d[11]
        ^ d[12]
        ^ d[13]
        ^ d[14]
        ^ d[15]
        ^ d[18]
        ^ d[19]
        ^ d[21],
        previous_d30
        ^ d[1]
        ^ d[3]
        ^ d[4]
        ^ d[5]
        ^ d[7]
        ^ d[8]
        ^ d[12]
        ^ d[13]
        ^ d[14]
        ^ d[15]
        ^ d[16]
        ^ d[19]
        ^ d[20]
        ^ d[22],
        previous_d30
        ^ d[0]
        ^ d[2]
        ^ d[4]
        ^ d[5]
        ^ d[6]
        ^ d[8]
        ^ d[9]
        ^ d[13]
        ^ d[14]
        ^ d[15]
        ^ d[16]
        ^ d[17]
        ^ d[20]
        ^ d[21]
        ^ d[23],
        previous_d29
        ^ d[2]
        ^ d[4]
        ^ d[5]
        ^ d[7]
        ^ d[8]
        ^ d[9]
        ^ d[10]
        ^ d[12]
        ^ d[14]
        ^ d[18]
        ^ d[21]
        ^ d[22]
        ^ d[23],
    ]


def make_raw_word(data: list[int], previous_d29: int, previous_d30: int) -> list[int]:
    parity = reference_parity(data, previous_d29, previous_d30)
    return [bit ^ previous_d30 for bit in data] + parity


def test_navigation_invert_binary_bits_returns_complement() -> None:
    assert Navigation()._invert([0, 1, 1, 0]) == [1, 0, 0, 1]


@pytest.mark.parametrize("bits", [[], [0], [1] * 30])
def test_navigation_invert_boundary_binary_inputs_returns_complement(
    bits: list[int],
) -> None:
    assert Navigation()._invert(bits) == [1 - bit for bit in bits]


@pytest.mark.parametrize("bits", [[2], [True], [-1], [0, 1, 0.0]])
def test_navigation_invert_nonbinary_input_raises(bits: list[object]) -> None:
    with pytest.raises(ValueError, match="binary"):
        Navigation()._invert(bits)  # type: ignore[arg-type]


def test_navigation_gps_parity_randomized_reference_matches() -> None:
    rng = random.Random(20261105)
    navigation = Navigation()
    for _ in range(100):
        data = [rng.randrange(2) for _ in range(24)]
        previous_d29, previous_d30 = rng.randrange(2), rng.randrange(2)
        assert navigation._gps_parity(
            data, previous_d29, previous_d30
        ) == reference_parity(data, previous_d29, previous_d30)


@pytest.mark.parametrize("data", [[], [0] * 23, [0] * 25])
def test_navigation_gps_parity_wrong_data_length_raises(data: list[int]) -> None:
    with pytest.raises(ValueError, match="24 bits"):
        Navigation()._gps_parity(data, 0, 0)


def test_navigation_decode_word_valid_parity_is_accepted() -> None:
    data = [int(bit) for bit in "101100100011010010101011"]
    raw_word = make_raw_word(data, previous_d29=1, previous_d30=0)

    assert Navigation().decode_word(raw_word, 1, 0) == (True, data)


def test_navigation_decode_word_single_parity_bit_error_is_rejected() -> None:
    data = [0] * 24
    raw_word = make_raw_word(data, 1, 1)
    raw_word[29] ^= 1

    assert Navigation().check_word(raw_word, 1, 1) is False


@pytest.mark.parametrize("raw_word", [[], [0] * 29, [0] * 31, [0] * 30 + [1]])
def test_navigation_decode_word_wrong_length_raises(raw_word: list[int]) -> None:
    with pytest.raises(ValueError, match="30 bits"):
        Navigation().decode_word(raw_word, 0, 0)


@pytest.mark.parametrize("raw_word", [[0] * 29 + [2], [0] * 29 + [True]])
def test_navigation_decode_word_nonbinary_input_raises(raw_word: list[object]) -> None:
    with pytest.raises(ValueError, match="binary"):
        Navigation().decode_word(raw_word, 0, 0)  # type: ignore[arg-type]


def test_navigation_decode_word_invalid_previous_parity_raises() -> None:
    with pytest.raises(ValueError, match="binary"):
        Navigation().decode_word([0] * 30, 2, 0)


def test_navigation_process_millisecond_running_median_gates_low_power() -> None:
    navigation = Navigation()
    navigation.process_millisecond(1, 10.0, 100.0, 0)
    navigation.process_millisecond(1, 10.0, 100.0, 1)
    navigation.process_millisecond(1, 10.0, 1.0, 2)

    assert navigation.prompt_signs == [1, 1]
    assert list(navigation.prompt_power_history) == [100.0, 100.0, 1.0]


def test_navigation_process_millisecond_uses_median_not_maximum() -> None:
    navigation = Navigation()
    navigation.prompt_power_history.extend([100.0, 10.0, 20.0])

    navigation.process_millisecond(1, 10.0, 30.0, 0)

    # Median([10, 20, 30, 100]) is 25; 30 is valid. A maximum threshold
    # would incorrectly reject this sample.
    assert navigation.prompt_signs == [1]


def test_navigation_process_millisecond_power_history_is_bounded() -> None:
    navigation = Navigation()
    for millisecond in range(200):
        navigation.process_millisecond(1, 1.0, float(millisecond + 1), millisecond)

    assert len(navigation.prompt_power_history) == 101
    assert list(navigation.prompt_power_history) == [
        float(value) for value in range(100, 201)
    ]


@pytest.mark.parametrize("millisecond", [-1, True, 1.5])
def test_navigation_process_millisecond_invalid_index_raises(
    millisecond: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Navigation().process_millisecond(1, 1.0, 1.0, millisecond)  # type: ignore[arg-type]


@pytest.mark.parametrize("prompt_value", [float("nan"), float("inf"), -float("inf")])
def test_navigation_process_millisecond_nonfinite_prompt_raises(
    prompt_value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        Navigation().process_millisecond(1, prompt_value, 1.0, 0)


def test_navigation_process_millisecond_locks_at_first_millisecond_after_boundary() -> (
    None
):
    navigation = Navigation()
    for millisecond in range(1000):
        navigation.process_millisecond(1, 1.0, 100.0, millisecond)

    assert navigation.bit_locked is False
    navigation.process_millisecond(1, 1.0, 100.0, 1000)
    assert navigation.bit_locked is True


def test_navigation_process_millisecond_does_not_emit_bit_with_missing_symbol_data() -> (
    None
):
    navigation = Navigation()
    navigation.bit_locked = True
    navigation.transition_bit = 19
    navigation.prompt_power_history.extend([100.0] * 21)

    for millisecond in range(20):
        navigation.process_millisecond(1, 1.0, 0.0, millisecond)

    assert navigation.navigation_bits == []
    assert navigation.bit_ready is False


def test_navigation_process_millisecond_result_reports_decoded_bit_readiness() -> None:
    navigation = Navigation()
    navigation.bit_locked = True
    navigation.transition_bit = 19

    for millisecond in range(20):
        result = navigation.process_millisecond(1, 1.0, 100.0, millisecond)

    assert result["bit"] == 1
    assert result["bit_ready"] is True
    assert navigation.navigation_bits == [1]


def test_navigation_process_millisecond_normal_preamble_is_not_overwritten() -> None:
    navigation = Navigation()
    navigation.navigation_bits = (
        [1, 0, 0, 0, 1, 0, 1, 1] + [0, 0] + [0, 1, 1, 1, 0, 1, 0, 0] + [0] * 12
    )

    navigation.process_millisecond(1, 1.0, 100.0, 0)

    assert navigation.preamble_locked is True
    assert navigation.costas_inverted is False
    assert navigation.preamble_index == 0
