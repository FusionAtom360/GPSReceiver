"""Synthetic GPS signal generation for deterministic tests."""

import math
import random


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
        prn_numbers: list[int], # Expected: 1 to 32
        doppler_shifts: list[float], #Expected: -5000 to 5000
        code_phases: list[float], # Expected: 0.0 to 1023.0
        carrier_to_noise_ratios: list[float], # Expected: 0.0 to 50.0
    ):
        self.prn_numbers = prn_numbers
        self.doppler_shifts = doppler_shifts
        self.code_phases = code_phases
        self.carrier_to_noise_ratios = carrier_to_noise_ratios
        self.sampling_rate = 1.023e6
        
        # Pre-generate the full 1,023-chip Gold code sequence for each requested satellite
        self.satellite_codes = []
        for prn in prn_numbers:
            # Assumes GPSPRNGenerator handles G2 tap selection based on the PRN number
            generator = GPSPRNGenerator(prn)
            self.satellite_codes.append(generator.generate_code())

    def generate_baseband_samples(self, num_samples: int = 1023) -> list[complex]:
        iq_samples = []
        
        signal_amplitudes = []
        for cn0 in self.carrier_to_noise_ratios:
            linear_ratio = 10.0 ** (cn0 / 10.0)
            amplitude = math.sqrt(linear_ratio / self.sampling_rate)
            signal_amplitudes.append(amplitude)

        for n in range(num_samples):
            t = n / self.sampling_rate
            combined_sample = complex(0.0, 0.0)
            
            for i, code in enumerate(self.satellite_codes):
                effective_chip_index = (n + self.code_phases[i]) % 1023
                chip_idx = int(effective_chip_index) % 1023
                
                chip = code[chip_idx]
                voltage = 1.0 if chip == 1 else -1.0
                
                scaled_voltage = voltage * signal_amplitudes[i]
                
                doppler_hz = self.doppler_shifts[i]
                phase = 2.0 * math.pi * doppler_hz * t
                carrier_i = math.cos(phase)
                carrier_q = math.sin(phase)
                
                sat_i = scaled_voltage * carrier_i
                sat_q = scaled_voltage * carrier_q
                
                combined_sample += complex(sat_i, sat_q)
            
            noise_i = random.gauss(0.0, 1.0)
            noise_q = random.gauss(0.0, 1.0)
            noise_sample = complex(noise_i, noise_q) * 0.707
            
            iq_samples.append(combined_sample + noise_sample)
            
        return iq_samples

if __name__ == "__main__":
    # Example usage
    prn_numbers = [1, 2, 3]  # PRN numbers for the channels
    doppler_shifts = [2000.0, -3000.0, 1000]  # Doppler shifts for each channel
    code_phases = [500.0, 0.0, 20.0]  # Code phases for each channel
    carrier_to_noise_ratios = [30.0, 30.0, 30.0]  # C/N ratios for each channel

    synthetic_channel = SyntheticGPSChannel(
        prn_numbers, doppler_shifts, code_phases, carrier_to_noise_ratios
    )
    baseband_samples = synthetic_channel.generate_baseband_samples()
    print(baseband_samples)
