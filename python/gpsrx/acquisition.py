"""FFT-based coarse acquisition."""

import math
from collections.abc import Sequence
from functools import lru_cache
import numpy as np
from gpsrx.synthetic import GPSPRNGenerator


@lru_cache(maxsize=64)
def _sampled_code_fft(prn_number: int, sampling_rate: float) -> np.ndarray:
    generator = GPSPRNGenerator(prn_number)
    raw_code = generator.generate_code()
    bipolar_code = 2.0 * np.asarray(raw_code, dtype=np.float64) - 1.0
    samples_per_code = int(round(sampling_rate * 0.001))
    sample_numbers = np.arange(samples_per_code, dtype=np.float64)
    chip_indices = (
        np.floor(sample_numbers * 1.023e6 / sampling_rate).astype(np.int64) % 1023
    )
    sampled_code = bipolar_code[chip_indices]
    return np.conj(np.fft.fft(sampled_code))


class FFTAcquisition:
    def __init__(
        self,
        prn_number: int,
        sampling_rate: float = 1.023e6,
        detection_threshold: float = 2.5,
        noncoherent_ms: int = 1,
        code_phase_exclusion_chips: float = 1.0,
    ):
        if not isinstance(prn_number, int) or isinstance(prn_number, bool):
            raise TypeError("prn_number must be an integer")
        if not 1 <= prn_number <= 32:
            raise ValueError("prn_number must be between 1 and 32")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be finite and positive")
        if int(round(sampling_rate * 0.001)) < 1:
            raise ValueError(
                "sampling_rate must produce at least one sample per code period"
            )
        if not math.isfinite(detection_threshold) or detection_threshold <= 0.0:
            raise ValueError("detection_threshold must be finite and positive")
        if not isinstance(noncoherent_ms, int) or isinstance(noncoherent_ms, bool):
            raise TypeError("noncoherent_ms must be an integer")
        if noncoherent_ms < 1:
            raise ValueError("noncoherent_ms must be positive")
        if (
            not math.isfinite(code_phase_exclusion_chips)
            or code_phase_exclusion_chips < 0.0
        ):
            raise ValueError(
                "code_phase_exclusion_chips must be finite and non-negative"
            )

        self.prn_number = prn_number
        self.sampling_rate = float(sampling_rate)
        self.chip_rate = 1.023e6
        self.samples_per_code = int(round(self.sampling_rate * 0.001))
        self.noncoherent_ms = noncoherent_ms
        self.detection_threshold = float(detection_threshold)
        self.code_phase_exclusion_chips = float(code_phase_exclusion_chips)
        self.code_fft_conj = _sampled_code_fft(prn_number, self.sampling_rate)

    def acquire(
        self,
        signal_block: np.ndarray,
        doppler_bins: Sequence[float],
        sample_offset: int = 0,
    ) -> dict[str, float | int | bool]:
        if not isinstance(sample_offset, int) or isinstance(sample_offset, bool):
            raise TypeError("sample_offset must be an integer")
        if sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        signal = np.asarray(signal_block)
        if signal.ndim != 1:
            raise ValueError("signal_block must be one-dimensional")
        if not np.issubdtype(signal.dtype, np.number):
            raise TypeError("signal_block must contain numeric samples")
        if not np.isfinite(signal).all():
            raise ValueError("signal_block must contain only finite samples")

        bins = np.asarray(list(doppler_bins), dtype=np.float64)
        if bins.ndim != 1 or bins.size == 0:
            raise ValueError(
                "doppler_bins must be a non-empty one-dimensional sequence"
            )
        if not np.isfinite(bins).all():
            raise ValueError("doppler_bins must contain only finite values")

        required_samples = self.samples_per_code * self.noncoherent_ms
        if signal.size < sample_offset + required_samples:
            raise ValueError(
                f"Signal block length must be at least {sample_offset + required_samples} samples "
                f"({self.noncoherent_ms} ms)."
            )

        signal = np.asarray(
            signal[sample_offset : sample_offset + required_samples],
            dtype=np.complex128,
        )
        sample_numbers = np.arange(self.samples_per_code, dtype=np.float64)
        segment_scores = np.zeros((bins.size, self.samples_per_code), dtype=np.float64)

        for segment in range(self.noncoherent_ms):
            start = segment * self.samples_per_code
            stop = start + self.samples_per_code
            segment_signal = signal[start:stop]
            segment_time = (sample_numbers + start + sample_offset) / self.sampling_rate
            carrier_wipe = np.exp(-1j * 2.0 * np.pi * bins[:, None] * segment_time)
            correlations = np.fft.ifft(
                np.fft.fft(segment_signal[None, :] * carrier_wipe, axis=1)
                * self.code_fft_conj[None, :],
                axis=1,
            )
            segment_scores += np.abs(correlations) ** 2

        best_bin_index, best_code_phase_sample = np.unravel_index(
            np.argmax(segment_scores), segment_scores.shape
        )
        best_score = float(segment_scores[best_bin_index, best_code_phase_sample])
        exclusion_samples = int(
            math.ceil(
                self.code_phase_exclusion_chips * self.sampling_rate / self.chip_rate
            )
        )
        second_peak_score = 0.0
        for code_phase_sample, score in enumerate(segment_scores[best_bin_index]):
            distance = abs(code_phase_sample - best_code_phase_sample)
            circular_distance = min(distance, self.samples_per_code - distance)
            if circular_distance > exclusion_samples:
                second_peak_score = max(second_peak_score, float(score))

        metric = (
            math.inf if second_peak_score == 0.0 else best_score / second_peak_score
        )
        detected = bool(best_score > 0.0 and metric >= self.detection_threshold)
        if not detected:
            best_score = 0.0
        code_phase_sample = (-int(best_code_phase_sample)) % self.samples_per_code

        return {
            "prn_number": self.prn_number,
            "detected": detected,
            "peak_magnitude": math.sqrt(best_score),
            "peak_metric": float(metric),
            "doppler_shift": float(bins[best_bin_index]) if detected else 0.0,
            "code_phase_samples": code_phase_sample if detected else 0,
            "code_phase_chips": (
                float(code_phase_sample * self.chip_rate / self.sampling_rate)
                if detected
                else 0.0
            ),
        }

if __name__ == "__main__":
    from gpsrx.synthetic import SyntheticGPSChannel
    # 1. Setup a synthetic channel containing PRN 12 with a known Doppler shift of +1250 Hz
    prn_numbers = [12]
    doppler_shifts = [1250.0]
    code_phases = [450.2]  # Initial code phase offset in chips
    cn0 = [42.0]

    channel = SyntheticGPSChannel(
        prn_numbers=prn_numbers,
        doppler_shifts=doppler_shifts,
        code_phases=code_phases,
        carrier_to_noise_ratios=cn0,
        sampling_rate=1.023e6
    )

    # 2. Generate 1 millisecond of IQ samples (1023 samples at 1.023 MHz)
    signal_block = channel.generate_baseband_samples(num_samples=1023)

    # 3. Initialize the FFT acquisition engine for PRN 12
    acq = FFTAcquisition(prn_number=12, sampling_rate=1.023e6)

    # 4. Define a Doppler search grid from -5000 Hz to 5000 Hz in 500 Hz steps
    doppler_grid = np.arange(-5000.0, 5001.0, 500.0)

    # 5. Run acquisition search
    result = acq.acquire(signal_block, doppler_grid)

    print("Acquisition Results:")
    print(f"  Best Doppler Shift : {result['doppler_shift']} Hz")
    print(f"  Estimated Code Phase: {result['code_phase_chips']:.2f} chips")
    print(f"  Correlation Peak   : {result['peak_magnitude']:.2f}")
