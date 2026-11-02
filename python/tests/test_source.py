import math
import random
import struct
from pathlib import Path

import pytest

from gpsrx.source import IQSource


def write_iq_file(path: Path, samples: list[tuple[float, float]]) -> None:
    packed_samples = (
        struct.pack("<ff", i_sample, q_sample) for i_sample, q_sample in samples
    )
    path.write_bytes(b"".join(packed_samples))


def test_iq_source_get_samples_typical_decodes_little_endian_iq(tmp_path: Path) -> None:
    write_iq_file(tmp_path / "samples.bin", [(1.25, -2.5), (-3.0, 4.75)])

    with IQSource(2_000_000, str(tmp_path / "samples.bin")) as source:
        assert source.get_samples(2) == [complex(1.25, -2.5), complex(-3.0, 4.75)]
        assert source.current_sample_index == 2
        assert source.sampling_rate == 2_000_000


def test_iq_source_get_samples_zero_returns_empty_without_advancing(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    with IQSource(1.0, str(path)) as source:
        assert source.get_samples(0) == []
        assert source.current_sample_index == 0


def test_iq_source_get_samples_eof_returns_available_complete_samples(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    write_iq_file(path, [(1.0, 2.0)])

    with IQSource(1.0, str(path)) as source:
        assert source.get_samples(2) == [complex(1.0, 2.0)]
        assert source.current_sample_index == 1
        assert source.get_samples(1) == []
        assert source.current_sample_index == 1


def test_iq_source_get_samples_partial_record_raises(tmp_path: Path) -> None:
    path = tmp_path / "partial.bin"
    path.write_bytes(struct.pack("<ff", 1.0, 2.0) + b"\x00")

    with IQSource(1.0, str(path)) as source:
        with pytest.raises(ValueError, match="partial sample record"):
            source.get_samples(2)
        assert source.current_sample_index == 0


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, float("nan"), float("inf")])
def test_iq_source_constructor_invalid_sampling_rate_raises(
    tmp_path: Path, sampling_rate: float
) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="finite and positive"):
        IQSource(sampling_rate, str(path))


@pytest.mark.parametrize("sampling_rate", [True, "1000000", None])
def test_iq_source_constructor_non_numeric_sampling_rate_raises(
    tmp_path: Path, sampling_rate: object
) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    with pytest.raises(TypeError, match="real number"):
        IQSource(sampling_rate, str(path))  # type: ignore[arg-type]


@pytest.mark.parametrize("num_samples", [-1, -100])
def test_iq_source_get_samples_negative_count_raises(
    tmp_path: Path, num_samples: int
) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(struct.pack("<ff", 1.0, 2.0))

    with IQSource(1.0, str(path)) as source:
        with pytest.raises(ValueError, match="num_samples must be non-negative"):
            source.get_samples(num_samples)
        assert source.current_sample_index == 0


@pytest.mark.parametrize("num_samples", [True, 1.5, "1", None])
def test_iq_source_get_samples_non_integer_count_raises(
    tmp_path: Path, num_samples: object
) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    with IQSource(1.0, str(path)) as source:
        with pytest.raises(TypeError, match="integer"):
            source.get_samples(num_samples)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_component", [math.nan, math.inf, -math.inf])
def test_iq_source_get_samples_non_finite_component_raises(
    tmp_path: Path, bad_component: float
) -> None:
    path = tmp_path / "non_finite.bin"
    write_iq_file(path, [(bad_component, 0.0)])

    with IQSource(1.0, str(path)) as source:
        with pytest.raises(ValueError, match="finite"):
            source.get_samples(1)
        assert source.current_sample_index == 0


def test_iq_source_context_manager_closes_file(tmp_path: Path) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    with IQSource(1.0, str(path)) as source:
        file_handle = source._file_handle
        assert not file_handle.closed
    assert file_handle.closed


def test_iq_source_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    source = IQSource(1.0, str(path))
    source.close()
    source.close()


def test_iq_source_get_samples_after_close_raises(tmp_path: Path) -> None:
    path = tmp_path / "samples.bin"
    path.write_bytes(b"")

    source = IQSource(1.0, str(path))
    source.close()

    with pytest.raises(ValueError, match="closed"):
        source.get_samples(0)


def test_iq_source_get_samples_randomized_chunking_preserves_order(tmp_path: Path) -> None:
    rng = random.Random(20261101)
    expected = [
        (rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0))
        for _ in range(128)
    ]
    path = tmp_path / "random.bin"
    write_iq_file(path, expected)

    # The expected values are the independently packed float32 values, not the
    # original Python doubles. This mirrors the file format's exact precision.
    expected_complex = [
        complex(i_sample, q_sample)
        for i_sample, q_sample in struct.iter_unpack(
            "<ff", path.read_bytes()
        )
    ]
    actual: list[complex] = []
    with IQSource(1.0, str(path)) as source:
        remaining = len(expected_complex)
        while remaining:
            chunk_size = min(remaining, rng.randint(1, 17))
            actual.extend(source.get_samples(chunk_size))
            remaining -= chunk_size

    assert actual == expected_complex
