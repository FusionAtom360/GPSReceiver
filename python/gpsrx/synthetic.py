"""Synthetic GPS signal generation for deterministic tests."""

import math
import random
from collections.abc import Sequence

import numpy as np

GPS_G2_TAPS = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}


class GPSPRNGenerator:
    def __init__(self, prn_number: int):
        if not isinstance(prn_number, int) or isinstance(prn_number, bool):
            raise TypeError("prn_number must be an integer")
        if prn_number not in GPS_G2_TAPS:
            raise ValueError("prn_number must be between 1 and 32")
        g2_tap1, g2_tap2 = GPS_G2_TAPS[prn_number]
        self.g1 = [1] * 10
        self.g2 = [1] * 10
        self.tap1 = g2_tap1 - 1
        self.tap2 = g2_tap2 - 1

    def clock(self) -> int:
        g1_feedback = self.g1[2] ^ self.g1[9]
        g2_feedback = (
            self.g2[1] ^ self.g2[2] ^ self.g2[5] ^ self.g2[7] ^ self.g2[8] ^ self.g2[9]
        )

        g1_out = self.g1[9]
        g2_out = self.g2[self.tap1] ^ self.g2[self.tap2]

        self.g1 = [g1_feedback] + self.g1[:9]
        self.g2 = [g2_feedback] + self.g2[:9]

        return g1_out ^ g2_out

    def generate_code(self) -> list[int]:
        return [self.clock() for _ in range(1023)]


class SyntheticGPSChannel:
    def __init__(
        self,
        prn_numbers: Sequence[int],
        doppler_shifts: Sequence[float],
        code_phases: Sequence[float],
        carrier_to_noise_ratios: Sequence[float],
        sampling_rate: float = 1.023e6,
        navigation_data: Sequence[int] | None = None,
        rng: random.Random | None = None,
    ):
        lengths = {
            len(prn_numbers),
            len(doppler_shifts),
            len(code_phases),
            len(carrier_to_noise_ratios),
        }
        if len(lengths) != 1:
            raise ValueError("all satellite inputs must have equal lengths")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0.0:
            raise ValueError("sampling_rate must be finite and positive")

        self.prn_numbers = list(prn_numbers)
        self.doppler_shifts = list(doppler_shifts)
        self.code_phases = []
        self.carrier_to_noise_ratios = list(carrier_to_noise_ratios)
        for prn, doppler, code_phase, cn0 in zip(
            self.prn_numbers,
            self.doppler_shifts,
            code_phases,
            self.carrier_to_noise_ratios,
            strict=True,
        ):
            GPSPRNGenerator(prn)
            if not math.isfinite(doppler) or not -5000.0 <= doppler <= 5000.0:
                raise ValueError(
                    f"doppler shift for PRN {prn} must be finite and in [-5000, 5000]"
                )
            if not math.isfinite(code_phase):
                raise ValueError(f"code phase for PRN {prn} must be finite")
            if not math.isfinite(cn0) or not 0.0 <= cn0 <= 50.0:
                raise ValueError(f"C/N0 for PRN {prn} must be finite and in [0, 50]")
            self.code_phases.append(code_phase % 1023.0)

        self.navigation_data: list[int] | None
        if navigation_data is not None:
            if not navigation_data:
                raise ValueError("navigation_data must not be empty")
            if any(
                not isinstance(bit, int) or isinstance(bit, bool) or bit not in (0, 1)
                for bit in navigation_data
            ):
                raise ValueError("navigation_data bits must be 0 or 1")
            self.navigation_data = [1 if bit == 1 else -1 for bit in navigation_data]
        else:
            self.navigation_data = None

        self.sampling_rate = float(sampling_rate)
        self.rng = rng if rng is not None else random.Random()
        self.sample_index = 0
        self.chip_rate = 1.023e6

        self.satellite_codes = []
        for prn in prn_numbers:
            generator = GPSPRNGenerator(prn)
            self.satellite_codes.append(generator.generate_code())

    def generate_baseband_samples(self, num_samples: int = 1023) -> list[complex]:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise TypeError("num_samples must be an integer")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")

        sample_numbers = self.sample_index + np.arange(num_samples, dtype=np.float64)
        times = sample_numbers / self.sampling_rate
        combined_samples = np.zeros(num_samples, dtype=np.complex128)
        for i, code in enumerate(self.satellite_codes):
            code_phase = (
                self.code_phases[i]
                + sample_numbers * self.chip_rate / self.sampling_rate
            )
            chip_indices = np.floor(code_phase).astype(np.int64) % 1023
            chips = 2.0 * np.asarray(code, dtype=np.float64)[chip_indices] - 1.0
            if self.navigation_data is not None:
                bit_indices = np.floor(
                    sample_numbers * self.chip_rate / self.sampling_rate / (20 * 1023)
                ).astype(np.int64)
                nav_bits = np.asarray(self.navigation_data, dtype=np.float64)
                chips *= nav_bits[bit_indices % len(nav_bits)]

            linear_ratio = 10.0 ** (self.carrier_to_noise_ratios[i] / 10.0)
            amplitude = math.sqrt(linear_ratio / self.sampling_rate)
            phase = 2.0 * math.pi * self.doppler_shifts[i] * times
            combined_samples += amplitude * chips * np.exp(1j * phase)

        noise = (
            np.array(
                [
                    complex(self.rng.gauss(0.0, 1.0), self.rng.gauss(0.0, 1.0))
                    for _ in range(num_samples)
                ],
                dtype=np.complex128,
            )
            * 0.707
        )
        self.sample_index += num_samples
        return [complex(value) for value in combined_samples + noise]


if __name__ == "__main__":
    # 1. Choose 3 satellites (PRN 1, 15, and 32)
    prn_numbers = [1, 15, 32]

    # 2. Assign realistic Doppler shifts (in Hz, within [-5000, 5000])
    doppler_shifts = [1200.5, -450.0, 2300.2]

    # 3. Assign initial code phases (in chips, between 0 and 1023)
    code_phases = [10.5, 512.0, 800.1]

    # 4. Assign Carrier-to-Noise ratios (C/N0 in dB-Hz, within [0, 50])
    carrier_to_noise_ratios = [40.0, 38.5, 42.0]

    # 5. Provide the binary navigation sequence "10001011"
    navigation_data = [1, 0, 0, 0, 1, 0, 1, 1]

    # 6. Initialize the channel simulator
    channel = SyntheticGPSChannel(
        prn_numbers=prn_numbers,
        doppler_shifts=doppler_shifts,
        code_phases=code_phases,
        carrier_to_noise_ratios=carrier_to_noise_ratios,
        sampling_rate=1.023e6,  # 1 sample per chip
        navigation_data=navigation_data,
    )

    # 7. Generate enough samples to transmit the full 8-bit sequence 
    # (8 bits * 20 ms/bit * 1023 samples/ms = 163,680 samples)
    num_samples = 8 * 20 * 1023
    iq_samples = channel.generate_baseband_samples(num_samples=num_samples)

    print(f"Generated {len(iq_samples)} complex baseband IQ samples.")
    print(f"First 5 IQ samples: {iq_samples[:5]}")
    
    with open("synthetic_gps_samples.txt", "w") as f:
        for sample in iq_samples:
            f.write(f"{sample.real:.6f} \t {sample.imag:.6f}j\n")
