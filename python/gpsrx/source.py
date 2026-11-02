"""IQ file sources and sample conversion."""

import math
import struct


class IQSource:
    def __init__(self, sampling_rate: float, file_path: str):
        if isinstance(sampling_rate, bool) or not isinstance(sampling_rate, (int, float)):
            raise TypeError("sampling_rate must be a real number")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be finite and positive")
        self.sampling_rate = sampling_rate
        self.file_path = file_path
        self.current_sample_index = 0
        self._file_handle = open(file_path, "rb")

    def __enter__(self) -> "IQSource":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        self._file_handle.close()

    def get_samples(self, num_samples: int) -> list[complex]:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise TypeError("num_samples must be an integer")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")

        data = self._file_handle.read(num_samples * 8)
        if len(data) % 8:
            raise ValueError("IQ data contains a partial sample record")

        samples = [
            complex(i_sample, q_sample)
            for i_sample, q_sample in struct.iter_unpack("<ff", data)
        ]
        if any(
            not math.isfinite(sample.real) or not math.isfinite(sample.imag)
            for sample in samples
        ):
            raise ValueError("IQ data must contain only finite samples")

        self.current_sample_index += len(samples)
        return samples


if __name__ == "__main__":
    with IQSource(1e6, "./data/captures/GPS-L1-2022-03-27.sigmf-data") as source:
        samples = source.get_samples(10)
        print(samples)
